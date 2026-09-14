/** @type {import('next').NextConfig} */
const nextConfig = {
  // The API base is read at build time so the same frontend can point at a
  // deployed backend without a code change.
  env: {
    API_BASE: process.env.API_BASE ?? "http://localhost:8000",
  },
};

export default nextConfig;
