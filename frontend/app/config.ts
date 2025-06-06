const isDevelopment = process.env.NODE_ENV === 'development';
export const API_URL = isDevelopment 
  ? 'http://localhost:8000'
  : 'https://the-ai-engineer-challenge-two.vercel.app'; 