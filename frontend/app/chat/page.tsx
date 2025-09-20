'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { API_URL } from '../config';

interface PDFStatus {
  pdf_uploaded: boolean;
  chunks_count: number;
}

export default function Chat() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [pdfStatus, setPdfStatus] = useState<PDFStatus>({ pdf_uploaded: false, chunks_count: 0 });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health and PDF status on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) {
          console.error('Health check failed:', response.status, response.statusText);
          setError('API is not responding. Please try again later.');
        }
      } catch (err) {
        console.error('Health check error:', err);
        setError('Cannot connect to API. Please check your connection.');
      }
    };

    const checkPdfStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/pdf-status`);
        if (response.ok) {
          const status = await response.json();
          setPdfStatus(status);
        }
      } catch (err) {
        console.error('PDF status check error:', err);
      }
    };

    checkHealth();
    checkPdfStatus();
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type !== 'application/pdf') {
        setError('Please select a PDF file.');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUploadPdf = async () => {
    if (!selectedFile || !apiKey.trim()) {
      setError('Please select a PDF file and enter your API key.');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('api_key', apiKey);

      console.log('Uploading PDF to:', `${API_URL}/api/upload-pdf`);
      const response = await fetch(`${API_URL}/api/upload-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('PDF Upload Error:', response.status, errorText);
        throw new Error(`Upload Error: ${response.status} ${errorText}`);
      }

      const result = await response.json();
      console.log('Upload successful:', result);
      
      // Update PDF status
      setPdfStatus({ pdf_uploaded: true, chunks_count: result.chunks_count });
      
      // Clear the file selection
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Add success message and summary to chat
      setMessages(prev => [...prev, { 
        role: 'system', 
        content: `PDF "${result.filename}" uploaded successfully! Created ${result.chunks_count} text chunks.\n\nDocument Summary:\n${result.summary}\n\nYou can now ask questions about the document.` 
      }]);

    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload PDF';
      setError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !apiKey.trim()) return;

    // Check if PDF is uploaded for RAG functionality
    if (!pdfStatus.pdf_uploaded) {
      setError('Please upload a PDF file first to enable chat functionality.');
      return;
    }

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);
    setError(null);

    try {
      console.log('Sending RAG chat request to:', `${API_URL}/api/rag-chat`);
      const response = await fetch(`${API_URL}/api/rag-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_message: userMessage,
          api_key: apiKey,
          k: 3, // Number of relevant chunks to retrieve
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('RAG Chat Error:', response.status, errorText);
        throw new Error(`Chat Error: ${response.status} ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      let assistantMessage = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = new TextDecoder().decode(value);
        assistantMessage += text;
        setMessages(prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage.role === 'assistant') {
            lastMessage.content = assistantMessage;
            return newMessages;
          } else {
            return [...newMessages, { role: 'assistant', content: assistantMessage }];
          }
        });
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      setError(errorMessage);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-4xl bg-white/80 backdrop-blur-md rounded-2xl shadow-2xl p-8 border border-blue-100">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-blue-700">Paper Summarizer & Analyzer</h1>
          <Link
            href="/"
            className="text-blue-500 hover:text-blue-600 transition-colors"
          >
            ← Back to Home
          </Link>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* API Key Input */}
        <div className="mb-6">
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your OpenAI API key"
            className="w-full p-3 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-blue-50 placeholder:text-blue-300"
          />
        </div>

        {/* PDF Upload Section */}
        <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-200">
          <h3 className="text-lg font-semibold text-purple-700 mb-3">Upload Research Paper (PDF)</h3>
          
          {/* PDF Status */}
          <div className="mb-3 flex items-center gap-2">
            <span className="text-sm text-gray-600">Status:</span>
            <span className={`px-2 py-1 rounded text-xs ${
              pdfStatus.pdf_uploaded ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
            }`}>
              {pdfStatus.pdf_uploaded 
                ? `PDF uploaded (${pdfStatus.chunks_count} chunks)` 
                : 'No PDF uploaded'
              }
            </span>
          </div>

          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="w-full p-2 border border-purple-200 rounded focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white"
              />
            </div>
            <button
              onClick={handleUploadPdf}
              disabled={!selectedFile || !apiKey || isUploading}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded font-bold shadow-md hover:from-purple-600 hover:to-pink-600 focus:outline-none focus:ring-2 focus:ring-purple-400 disabled:opacity-50 transition-all duration-150"
            >
              {isUploading ? 'Uploading...' : 'Upload PDF'}
            </button>
          </div>
          
          {selectedFile && (
            <div className="mt-2 text-sm text-purple-600">
              Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </div>

        {/* Chat Messages */}
        <div className="h-[400px] overflow-y-auto mb-6 p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-100 shadow-inner">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-32">
              Upload a research paper and start exploring its contents!
            </div>
          )}
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 p-3 rounded-xl max-w-[80%] shadow-sm ${
                message.role === 'user'
                  ? 'bg-blue-100 ml-auto text-right'
                  : message.role === 'system'
                  ? 'bg-emerald-50 mx-auto text-center border border-emerald-200'
                  : 'bg-indigo-50 mr-auto text-left border border-indigo-200'
              }`}
            >
              <span className="block text-xs font-semibold mb-1 text-indigo-600">
                {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Research Assistant'}
              </span>
              <p className="text-gray-800 whitespace-pre-line">{message.content}</p>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={pdfStatus.pdf_uploaded ? "Ask about methodology, references, pros/cons, or any aspect of the paper..." : "Upload a research paper to start analyzing"}
            className="flex-1 p-3 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white placeholder:text-blue-300"
            disabled={isLoading || !pdfStatus.pdf_uploaded}
          />
          <button
            type="submit"
            disabled={isLoading || !pdfStatus.pdf_uploaded}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-bold shadow-md hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 transition-all duration-150"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}