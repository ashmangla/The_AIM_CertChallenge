const isDevelopment = process.env.NODE_ENV === 'development';
export const API_URL = isDevelopment 
  ? 'http://localhost:8000'
  : process.env.NEXT_PUBLIC_API_URL || ''; 