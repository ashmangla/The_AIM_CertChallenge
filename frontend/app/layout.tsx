import './globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'HandyAssist - AI Appliance Assistant',
  description: 'Your AI-powered assistant for appliance manuals. Ask questions and get instant answers about how to use, maintain, and troubleshoot your appliances.',
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
