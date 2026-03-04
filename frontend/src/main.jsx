import React from 'react'
import ReactDOM from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import App from './App.jsx'
import './index.css'

const domain = import.meta.env.VITE_AUTH0_DOMAIN
const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID
const audience = import.meta.env.VITE_AUTH0_AUDIENCE

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={{
        redirect_uri: window.location.origin,
        ...(audience && { audience }), // Only include audience if defined
        scope: "openid profile email offline_access"
      }}
      cacheLocation="localstorage"  // Cache tokens in localStorage
      useRefreshTokens={true}  // Use refresh tokens
    >
      <App />
    </Auth0Provider>
  </React.StrictMode>,
)
