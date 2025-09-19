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
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">RAG-Enabled PDF Chat</h2>
          
          <p className="text-gray-600 mb-4">
            This application enables you to upload PDF documents and have intelligent conversations about their content.
            Using Retrieval-Augmented Generation (RAG), the AI can answer questions based specifically on the information 
            contained in your uploaded documents.
          </p>

          <h3 className="text-xl font-semibold text-gray-800 mb-3">How it Works</h3>
          <ol className="list-decimal list-inside text-gray-600 mb-4">
            <li>Upload a PDF document using the file upload interface</li>
            <li>The system extracts and chunks the text content</li>
            <li>Text chunks are converted to embeddings and stored in a vector database</li>
            <li>When you ask questions, relevant chunks are retrieved</li>
            <li>The AI answers based only on the retrieved document content</li>
          </ol>

          <h3 className="text-xl font-semibold text-gray-800 mb-3">Features</h3>
          <ul className="list-disc list-inside text-gray-600 mb-4">
            <li>PDF document upload and processing</li>
            <li>Intelligent text chunking and embedding</li>
            <li>Vector similarity search for relevant content</li>
            <li>Context-aware AI responses</li>
            <li>Real-time streaming responses</li>
            <li>Modern, responsive design</li>
            <li>Secure API key handling</li>
          </ul>

          <h3 className="text-xl font-semibold text-gray-800 mb-3">Technology Stack</h3>
          <ul className="list-disc list-inside text-gray-600 mb-4">
            <li>Frontend: Next.js, React, TypeScript, Tailwind CSS</li>
            <li>Backend: FastAPI, Python</li>
            <li>RAG: aimakerspace library with OpenAI embeddings</li>
            <li>PDF Processing: PyPDF2</li>
            <li>Vector Search: Numpy-based cosine similarity</li>
            <li>AI: OpenAI API (GPT-4o-mini)</li>
          </ul>

          <div className="mt-8 text-center">
            <Link
              href="/chat"
              className="inline-block px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-bold shadow-md hover:from-blue-600 hover:to-purple-600 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-150"
            >
              Try RAG Chat
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 