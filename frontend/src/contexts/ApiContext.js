import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';

const STATUS_POLL_MS = 3000;

// Create context
const ApiContext = createContext();

// Custom hook to use the API context
export const useApi = () => useContext(ApiContext);

// Provider component
export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isResourcesLoaded, setIsResourcesLoaded] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(null);
  const [numDocs, setNumDocs] = useState(5); // Default to 5 documents
  // Hungarian eval demo (evaluation/hu_legal/demo.py); null against the generic API
  const [demo, setDemo] = useState(null);
  const [evalQueries, setEvalQueries] = useState([]);
  const [extractor, setExtractor] = useState(null);

  // Add a ref to track document updates
  const documentUpdateTimeoutRef = useRef(null);
  
  // One status check; returns whether the server is ready. "Not up yet" (connection
  // refused: Vite's empty 500, nginx's 502) stays silent; an error the API itself reports
  // ({detail}, e.g. a failed RAG init) is shown in the banner.
  const refreshStatus = useCallback(async () => {
    try {
      const response = await axios.get('/api/status');
      const ready = response.data?.resources_loaded === true;
      setIsResourcesLoaded(ready);
      if (ready) setError(null);
      return ready;
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string') setError(detail);
      return false;
    }
  }, []);

  // Poll until ready: the API (and the demo server, which binds only after loading its
  // models) may start after the page. One request at a time, with no timeout, because the
  // generic API builds its models inside the first /api/status call.
  useEffect(() => {
    if (isResourcesLoaded) return undefined;
    let cancelled = false;
    let timer = null;
    const loop = async () => {
      const ready = await refreshStatus();
      if (!cancelled && !ready) timer = setTimeout(loop, STATUS_POLL_MS);
    };
    loop();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [isResourcesLoaded, refreshStatus]);

  // Detect the demo server once the API answers; the generic API has no /api/demo routes
  // (404 → stay null)
  useEffect(() => {
    if (!isResourcesLoaded) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const config = (await axios.get('/api/demo/config')).data;
        if (cancelled || !Array.isArray(config?.extractors) || !config.extractors.length) return;
        setDemo(config);
        setExtractor(config.default_extractor);
        const payload = (await axios.get('/api/demo/queries')).data;
        if (!cancelled) setEvalQueries(Array.isArray(payload?.queries) ? payload.queries : []);
      } catch (err) {
        if (err.response?.status !== 404) console.error('Demo config failed:', err);
      }
    })();
    return () => { cancelled = true; };
  }, [isResourcesLoaded]);

  // Submit a query; evalQuery is the eval-set query it came from, if any (for gold marks)
  const submitQuery = useCallback(async (question, evalQuery = null) => {
    if (!isResourcesLoaded) {
      setError('Resources not loaded. Please wait for the system to initialize.');
      return null;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Create an initial query result structure
    setCurrentQuery({
      question,
      evalQuery,
      extractor: demo ? extractor : null,
      documents: [],
      answer: null,
      structured_answer: null
    });
    
    try {
      // Use fetch for streaming instead of axios
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no'
        },
        body: JSON.stringify({
          question,
          num_docs: numDocs,
          ...(demo ? { extractor } : {})
        })
      });

      if (!response.ok) {
        let detail = response.statusText;
        try {
          detail = (await response.json()).detail || detail;
        } catch (e) {
          // body is not JSON; keep the status text
        }
        throw new Error(`Server responded with ${response.status}: ${detail}`);
      }
      
      const reader = response.body.getReader();
      let buffer = '';
      const decoder = new TextDecoder();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Split by newlines but keep the last potentially incomplete line
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep last line in buffer if incomplete
        
        // Process only complete lines
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            let data;
            
            // Handle both normal and double-escaped JSON (for Fly.io compatibility)
            try {
              // First try normal JSON parsing (works locally)
              data = JSON.parse(line);
            } catch (e) {
              // If that fails, check if it's double-escaped (Fly.io issue)
              if (line.startsWith('"') && line.endsWith('"')) {
                // Remove outer quotes and parse twice
                const unescaped = JSON.parse(line); // First parse removes escaping
                data = JSON.parse(unescaped);       // Second parse gets actual object
              } else {
                throw e; // Re-throw if neither format works
              }
            }
            
            if (data.error) {
              setError(data.error);
              continue;
            }
            
            switch (data.type) {
              case 'documents':
                // Update documents incrementally
                setCurrentQuery(prev => ({
                  ...prev,
                  documents: data.data
                }));
                break;
                
              case 'highlights':
                // Update documents with highlights
                setCurrentQuery(prev => {
                  // Create a map of document content to highlights
                  const highlightMap = {};
                  data.data.forEach(doc => {
                    highlightMap[doc.content] = doc.highlights || [];
                  });
                  
                  // Update each document with its highlights
                  const updatedDocs = prev.documents.map(doc => ({
                    ...doc,
                    highlights: highlightMap[doc.content] || []
                  }));
                  
                  return {
                    ...prev,
                    documents: updatedDocs
                  };
                });
                break;
                
              case 'answer':
                // Update with final answer
                setCurrentQuery(prev => ({
                  ...prev,
                  answer: data.data.answer,
                  structured_answer: data.data.structured_answer
                }));
                
                if (data.done) {
                  setIsLoading(false);
                }
                break;

              case 'progress':
                break;

              default:
                console.warn('Unknown response type:', data.type);
            }
          } catch (parseError) {
            console.error('Error parsing response:', parseError);
            console.error('Raw line:', line);
          }
        }
      }
      
      // Process any remaining buffer content
      if (buffer.trim()) {
        try {
          let data;
          
          // Handle both normal and double-escaped JSON (for Fly.io compatibility)
          try {
            data = JSON.parse(buffer);
          } catch (e) {
            if (buffer.startsWith('"') && buffer.endsWith('"')) {
              const unescaped = JSON.parse(buffer);
              data = JSON.parse(unescaped);
            } else {
              throw e;
            }
          }
          
          // Process final data if needed
          if (data.type === 'answer' && data.done) {
            setIsLoading(false);
          }
        } catch (parseError) {
          console.error('Error parsing final buffer:', parseError);
          console.error('Raw buffer:', buffer);
        }
      }
      
      return currentQuery;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isResourcesLoaded, numDocs, demo, extractor]);

  // Load resources with optional API key
  const loadResources = useCallback(async (apiKey = null) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const payload = apiKey ? { api_key: apiKey } : {};
      const response = await axios.post('/api/load-resources', payload);
      
      setIsResourcesLoaded(response.data.success);
      
      if (!response.data.success) {
        setError(response.data.message);
      }
      
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update number of documents to retrieve - keep this for internal use
  const updateNumDocs = useCallback((num) => {
    setNumDocs(num);
  }, []);

  // Reset current query
  const resetQuery = useCallback(() => {
    setCurrentQuery(null);
    setError(null);
  }, []);

  // Force UI updates when documents change
  useEffect(() => {
    if (currentQuery && currentQuery.documents && currentQuery.documents.length > 0 && isLoading) {
      // Clear any existing timeout
      if (documentUpdateTimeoutRef.current) {
        clearTimeout(documentUpdateTimeoutRef.current);
      }
      
      // Force a UI update after a short delay
      documentUpdateTimeoutRef.current = setTimeout(() => {
        // This is just to trigger a re-render
        setCurrentQuery(prev => ({...prev}));
      }, 50);
    }
    
    // Cleanup on unmount
    return () => {
      if (documentUpdateTimeoutRef.current) {
        clearTimeout(documentUpdateTimeoutRef.current);
      }
    };
  }, [currentQuery, isLoading]);

  // Value object
  const value = {
    isLoading,
    error,
    isResourcesLoaded,
    currentQuery,
    numDocs,
    submitQuery,
    loadResources,
    updateNumDocs,
    resetQuery,
    refreshStatus,
    demo,
    evalQueries,
    extractor,
    setExtractor,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

export default ApiContext; 