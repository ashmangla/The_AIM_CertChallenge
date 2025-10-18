'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { API_URL } from '../config';

interface AgentStatus {
  status: string;
  message: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentReady, setAgentReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health and agent status on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) {
          console.error('Health check failed:', response.status, response.statusText);
          setError('HandyAssist agent is not responding. Please try again later.');
          setAgentReady(false);
        } else {
          setAgentReady(true);
          // Add welcome message
          setMessages([{
            role: 'assistant',
            content: '👋 Hi! I\'m HandyAssist, your AI appliance manual assistant. I have access to your GE Fridge manual. Ask me anything about how to use, maintain, or troubleshoot your appliance!'
          }]);
        }
      } catch (err) {
        console.error('Health check error:', err);
        setError('Cannot connect to HandyAssist. Please check your connection.');
        setAgentReady(false);
      }
    };

    checkHealth();
  }, []);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !apiKey.trim()) {
      setError('Please enter your OpenAI API key and a message.');
      return;
    }

    if (!agentReady) {
      setError('HandyAssist agent is not ready. Please wait or refresh the page.');
      return;
    }

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);
    setError(null);

    try {
      console.log('Sending agent chat request to:', `${API_URL}/api/rag-chat-mixed-media`);
      const response = await fetch(`${API_URL}/api/rag-chat-mixed-media`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_message: userMessage,
          api_key: apiKey,
          k: 8,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Agent Chat Error:', response.status, errorText);
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
          <div>
            <h1 className="text-3xl font-bold text-blue-700">🔧 HandyAssist</h1>
            <p className="text-sm text-gray-600">Your AI Appliance Manual Assistant</p>
          </div>
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

        {/* Agent Status */}
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-gray-700">Agent Status:</span>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              agentReady ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
            }`}>
              {agentReady ? '✓ Ready to Help' : '⏳ Initializing...'}
            </span>
          </div>
          <p className="text-xs text-gray-600">
            I have access to your GE Fridge manual and can search the web for additional information.
          </p>
        </div>

        {/* API Key Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OpenAI API Key (required for chat)
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full p-3 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-blue-50 placeholder:text-blue-300"
          />
        </div>

        {/* Chat Messages */}
        <div className="h-[400px] overflow-y-auto mb-6 p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-100 shadow-inner">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 p-3 rounded-xl max-w-[80%] shadow-sm ${
                message.role === 'user'
                  ? 'bg-blue-100 ml-auto text-right'
                  : 'bg-white mr-auto text-left border border-blue-200'
              }`}
            >
              <span className="block text-xs font-semibold mb-1 text-blue-600">
                {message.role === 'user' ? 'You' : '🔧 HandyAssist'}
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
            placeholder={agentReady ? "Ask about your GE Fridge: ice maker, temperature, cleaning, troubleshooting..." : "Waiting for agent to initialize..."}
            className="flex-1 p-3 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white placeholder:text-blue-300"
            disabled={isLoading || !agentReady}
          />
          <button
            type="submit"
            disabled={isLoading || !agentReady || !apiKey.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-bold shadow-md hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 transition-all duration-150"
          >
            {isLoading ? '🤔 Thinking...' : 'Ask'}
          </button>
        </form>
      </div>
    </div>
  );
}