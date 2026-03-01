import { useAuth0 } from '@auth0/auth0-react'

export default function EmailVerificationRequired() {
  const { logout, user } = useAuth0()

  const handleResendEmail = async () => {
    // Auth0 doesn't have a direct API to resend verification email from frontend
    // User needs to go to their email or we can provide instructions
    alert('Please check your email inbox and spam folder for the verification email from Auth0. If you cannot find it, please contact support.')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Warning Icon */}
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
              <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              Email Verification Required
            </h1>
            <p className="text-gray-600">
              Please verify your email address to continue
            </p>
          </div>

          {/* User Email */}
          {user?.email && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <p className="text-sm text-gray-600 mb-1">Verification email sent to:</p>
              <p className="text-sm font-medium text-gray-900">{user.email}</p>
            </div>
          )}

          {/* Instructions */}
          <div className="space-y-3 mb-6">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-xs font-bold text-primary-600">1</span>
              </div>
              <p className="text-sm text-gray-700">
                Check your email inbox for a verification email from Auth0
              </p>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-xs font-bold text-primary-600">2</span>
              </div>
              <p className="text-sm text-gray-700">
                Click the verification link in the email
              </p>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-xs font-bold text-primary-600">3</span>
              </div>
              <p className="text-sm text-gray-700">
                Return here and refresh the page or log in again
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <button
              onClick={() => window.location.reload()}
              className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg hover:bg-primary-700 transition-colors font-medium"
            >
              I've Verified - Refresh Page
            </button>
            
            <button
              onClick={handleResendEmail}
              className="w-full bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-colors font-medium"
            >
              Didn't Receive Email?
            </button>
            
            <button
              onClick={() => logout({ returnTo: window.location.origin })}
              className="w-full text-gray-600 py-2 px-4 rounded-lg hover:bg-gray-50 transition-colors text-sm"
            >
              Log Out
            </button>
          </div>

          {/* Help Text */}
          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-xs text-yellow-800">
              <strong>Note:</strong> Check your spam folder if you don't see the email. 
              The verification link expires after 24 hours.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
