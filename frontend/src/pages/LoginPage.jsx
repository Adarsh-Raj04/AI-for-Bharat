import { useAuth0 } from "@auth0/auth0-react";

export default function LoginPage() {
  const { loginWithRedirect } = useAuth0();

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Logo/Icon */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-full mb-4">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              MedResearch AI
            </h1>
            <p className="text-gray-600">
              Your intelligent research assistant for life sciences
            </p>
          </div>

          {/* Features */}
          <div className="space-y-3 mb-8">
            <div className="flex items-start space-x-3">
              <svg
                className="w-5 h-5 text-primary-600 mt-0.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <p className="text-sm font-medium text-gray-900">
                  Summarize Research Papers
                </p>
                <p className="text-xs text-gray-500">
                  Get instant summaries from PubMed and clinical trials
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-3">
              <svg
                className="w-5 h-5 text-primary-600 mt-0.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <p className="text-sm font-medium text-gray-900">
                  Compare Drug Efficacy
                </p>
                <p className="text-xs text-gray-500">
                  Analyze and compare clinical trial results
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-3">
              <svg
                className="w-5 h-5 text-primary-600 mt-0.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <p className="text-sm font-medium text-gray-900">
                  Navigate Regulations
                </p>
                <p className="text-xs text-gray-500">
                  Access FDA and regulatory guidance
                </p>
              </div>
            </div>
          </div>

          {/* Login Button */}
          <button
            onClick={() =>
              loginWithRedirect({
                authorizationParams: {
                  audience: "https://medresearch-ai-api",
                  scope: "openid profile email offline_access",
                },
              })
            }
            className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg hover:bg-primary-700 transition-colors font-medium shadow-md hover:shadow-lg"
          >
            Log In to Get Started
          </button>

          {/* Disclaimer */}
          <p className="text-xs text-gray-500 text-center mt-6">
            ⚠️ For research purposes only. Not medical advice.
          </p>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-gray-600 mt-6">
          Built for AWS AI for Bharat Hackathon
        </p>
      </div>
    </div>
  );
}
