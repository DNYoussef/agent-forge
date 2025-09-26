/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      // Proxy pipeline API calls to backend
      {
        source: '/api/pipeline/:path*',
        destination: 'http://localhost:8000/api/pipeline/:path*',
      },
      // Proxy phase-specific API calls
      {
        source: '/api/phases/:path*',
        destination: 'http://localhost:8000/api/phases/:path*',
      },
      // Keep stats endpoint local (Next.js API route)
      // Don't proxy /api/stats - it's handled by Next.js
    ]
  },
  // Enable WebSocket support
  webpack: (config) => {
    config.externals.push({
      'bufferutil': 'bufferutil',
      'utf-8-validate': 'utf-8-validate',
    });
    return config;
  },
}

module.exports = nextConfig