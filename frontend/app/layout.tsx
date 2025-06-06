import './globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'AI Chat Interface',
  description: 'A modern AI chat interface powered by FastAPI and OpenAI',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-100 font-sans text-gray-900">
        <main className="flex flex-col items-center justify-center min-h-screen w-full">
          {children}
        </main>
      </body>
    </html>
  );
}
