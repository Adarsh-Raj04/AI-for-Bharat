# MedResearch AI - Frontend

React + Vite + Tailwind CSS frontend with Auth0 authentication.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Auth0 credentials
```

3. Start development server:
```bash
npm run dev
```

Frontend will be available at http://localhost:5173

## Build for Production

```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Page components
│   ├── services/       # API client
│   ├── App.jsx         # Main app component
│   └── main.jsx        # Entry point
├── public/             # Static assets
└── package.json
```

## Environment Variables

Required in `.env`:
- `VITE_AUTH0_DOMAIN` - Auth0 tenant domain
- `VITE_AUTH0_CLIENT_ID` - Auth0 application client ID
- `VITE_AUTH0_AUDIENCE` - Auth0 API identifier
- `VITE_API_BASE_URL` - Backend API URL (default: http://localhost:8000)

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Key Features

- Auth0 authentication with JWT tokens
- Real-time chat interface
- Chat history sidebar
- Citation display for AI responses
- Disclaimer modal
- Responsive design with Tailwind CSS
