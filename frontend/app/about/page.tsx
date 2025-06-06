'use client';

import Link from 'next/link';

export default function About() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-2xl bg-white/80 backdrop-blur-md rounded-2xl shadow-2xl p-8 border border-blue-100">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-blue-700">About</h1>
          <Link
            href="/"
            className="text-blue-500 hover:text-blue-600 transition-colors"
          >
            ← Back to Home
          </Link>
        </div>

        <div className="prose prose-blue max-w-none">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">AI Chat Interface</h2>
          
          <p className="text-gray-600 mb-4">
            This application provides a modern interface for interacting with OpenAI's powerful language models.
            Built with Next.js and FastAPI, it offers a seamless experience for AI-powered conversations.
          </p>

          <h3 className="text-xl font-semibold text-gray-800 mb-3">Features</h3>
          <ul className="list-disc list-inside text-gray-600 mb-4">
            <li>Real-time streaming responses</li>
            <li>Modern, responsive design</li>
            <li>Secure API key handling</li>
            <li>Health monitoring</li>
            <li>Easy-to-use interface</li>
          </ul>

          <h3 className="text-xl font-semibold text-gray-800 mb-3">Technology Stack</h3>
          <ul className="list-disc list-inside text-gray-600 mb-4">
            <li>Frontend: Next.js, React, TypeScript, Tailwind CSS</li>
            <li>Backend: FastAPI, Python</li>
            <li>AI: OpenAI API</li>
          </ul>

          <div className="mt-8 text-center">
            <Link
              href="/chat"
              className="inline-block px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-bold shadow-md hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-150"
            >
              Start Chatting
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 