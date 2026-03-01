import React from 'react';
import { AlertTriangle } from 'lucide-react';

const DisclaimerBanner = () => {
  return (
    <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <AlertTriangle className="h-5 w-5 text-yellow-400" />
        </div>
        <div className="ml-3">
          <p className="text-sm text-yellow-800 font-medium">
            <strong>Medical Disclaimer:</strong> This system is for research and informational purposes only. 
            It does not provide medical advice, diagnosis, or treatment recommendations. 
            Always consult a qualified healthcare professional for medical decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DisclaimerBanner;
