/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/intent',
        destination: process.env.NODE_ENV === 'development'
          ? 'http://localhost:8000/api/intent'
          : '/api/intent',
      },
    ];
  },
};

export default nextConfig;
