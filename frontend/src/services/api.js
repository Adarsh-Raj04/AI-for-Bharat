import axios from "axios";
const API_URL = import.meta.env.VITE_API_URL;

const api = axios.create({
  baseURL: API_URL + "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add auth token to all requests
api.interceptors.request.use(
  async (config) => {
    // Token will be added by the calling component using getAccessTokenSilently
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error("Authentication error - token may be invalid or expired");
      // Could trigger logout or token refresh here
    } else if (error.response?.status === 429) {
      console.error("Rate limit exceeded");
    }
    return Promise.reject(error);
  },
);

export default api;
