import { useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import api from '../services/api'

export default function DisclaimerModal({ isOpen, onClose }) {
  const [accepted, setAccepted] = useState(false)
  const [loading, setLoading] = useState(false)
  const { getAccessTokenSilently } = useAuth0()

  if (!isOpen) return null

  const handleAccept = async () => {
    if (!accepted) return

    setLoading(true)
    try {
      const token = await getAccessTokenSilently()
      await api.post('/auth/disclaimer/accept', {}, {
        headers: { Authorization: `Bearer ${token}` }
      })
      onClose()
    } catch (error) {
      console.error('Failed to accept disclaimer:', error)
      alert('Failed to save disclaimer acceptance. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Important Disclaimer
          </h2>
          
          <div className="prose prose-sm max-w-none mb-6">
            <p className="text-gray-700">
              Welcome to MedResearch AI. Before using this service, please read and accept the following terms:
            </p>

            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 my-4">
              <p className="font-semibold text-yellow-800">⚠️ Not Medical Advice</p>
              <p className="text-yellow-700 text-sm mt-1">
                This tool is for research and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
              </p>
            </div>

            <h3 className="text-lg font-semibold text-gray-900 mt-4 mb-2">Key Limitations:</h3>
            <ul className="list-disc pl-5 space-y-1 text-gray-700">
              <li>AI-generated responses may contain errors or inaccuracies</li>
              <li>Information should be independently verified before use</li>
              <li>Not intended for clinical decision-making</li>
              <li>Uses publicly available data sources only</li>
              <li>May not reflect the most current research</li>
            </ul>

            <h3 className="text-lg font-semibold text-gray-900 mt-4 mb-2">Responsible Use:</h3>
            <ul className="list-disc pl-5 space-y-1 text-gray-700">
              <li>Always consult qualified healthcare professionals for medical decisions</li>
              <li>Verify all information with primary sources</li>
              <li>Do not use for patient care or clinical practice</li>
              <li>Understand that AI can make mistakes</li>
            </ul>

            <p className="text-gray-700 mt-4">
              By accepting, you acknowledge that you understand these limitations and agree to use MedResearch AI responsibly.
            </p>
          </div>

          <div className="flex items-start mb-6">
            <input
              type="checkbox"
              id="accept-disclaimer"
              checked={accepted}
              onChange={(e) => setAccepted(e.target.checked)}
              className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="accept-disclaimer" className="ml-3 text-sm text-gray-700">
              I have read and understand the disclaimer. I agree to use MedResearch AI responsibly and for research purposes only.
            </label>
          </div>

          <div className="flex justify-end space-x-3">
            <button
              onClick={handleAccept}
              disabled={!accepted || loading}
              className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {loading ? 'Accepting...' : 'Accept & Continue'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
