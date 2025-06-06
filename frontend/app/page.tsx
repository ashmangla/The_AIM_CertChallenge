'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function Home() {
  const [apiStatus, setApiStatus] = useState<'loading' | 'healthy' | 'error'>('loading');

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/health');
        if (response.ok) {
          setApiStatus('healthy');
        } else {
          setApiStatus('error');
        }
      } catch (error) {
        setApiStatus('error');
      }
    };

    checkApiHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-2xl bg-white/80 backdrop-blur-md rounded-2xl shadow-2xl p-8 border border-blue-100">
        <h1 className="text-4xl font-extrabold text-center mb-6 text-blue-700 drop-shadow-lg tracking-tight">
          AI Chat Interface
        </h1>
        
        <div className="flex items-center justify-center mb-8">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              apiStatus === 'loading' ? 'bg-yellow-400 animate-pulse' :
              apiStatus === 'healthy' ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            <span className="text-sm text-gray-600">
              {apiStatus === 'loading' ? 'Checking API...' :
               apiStatus === 'healthy' ? 'API Connected' : 'API Error'}
            </span>
          </div>
        </div>

        <div className="space-y-4">
          <Link 
            href="/chat"
            className="block w-full p-4 text-center bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-bold shadow-md hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-150"
          >
            Start Chat
          </Link>
          
          <Link
            href="/about"
            className="block w-full p-4 text-center bg-white text-blue-600 border-2 border-blue-500 rounded-lg font-bold shadow-md hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-150"
          >
            About
          </Link>
        </div>

        <p className="mt-8 text-center text-gray-500 text-sm">
          Powered by FastAPI & OpenAI
        </p>
      </div>
    </div>
  );
} 