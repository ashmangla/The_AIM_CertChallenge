const isDevelopment = process.env.NODE_ENV === 'development';
const VERCEL_URL = process.env.NEXT_PUBLIC_VERCEL_URL;

export const API_URL = isDevelopment 
  ? 'http://localhost:8000'
  : VERCEL_URL 
    ? `https://${VERCEL_URL}/api` 
    : '/api';