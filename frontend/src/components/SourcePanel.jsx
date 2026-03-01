import React from 'react';
import { ExternalLink, FileText, AlertTriangle, CheckCircle } from 'lucide-react';

const SourcePanel = ({ documents, biasAnalysis, confidence }) => {
  if (!documents || documents.length === 0) {
    return null;
  }

  const getSourceIcon = (sourceType) => {
    return <FileText className="w-4 h-4" />;
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadge = (score) => {
    if (score >= 0.8) return 'bg-green-100 text-green-800';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Sources ({documents.length})
        </h3>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceBadge(confidence)}`}>
          {(confidence * 100).toFixed(0)}% Confidence
        </div>
      </div>

      {/* Bias Warning */}
      {biasAnalysis && biasAnalysis.has_bias && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-yellow-900 mb-1">
                Potential Bias Detected
              </p>
              {biasAnalysis.bias_flags && biasAnalysis.bias_flags.map((flag, idx) => (
                <p key={idx} className="text-sm text-yellow-800">
                  • {flag.message}
                </p>
              ))}
              {biasAnalysis.recommendations && biasAnalysis.recommendations.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-yellow-900 mb-1">Recommendations:</p>
                  {biasAnalysis.recommendations.map((rec, idx) => (
                    <p key={idx} className="text-xs text-yellow-800">
                      • {rec}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Source List */}
      <div className="space-y-3">
        {documents.map((doc, index) => (
          <div
            key={doc.id || index}
            className="p-3 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center space-x-2 flex-1">
                {getSourceIcon(doc.source_type)}
                <span className="text-xs font-medium text-gray-500 uppercase">
                  {doc.source_type}
                </span>
                {doc.publication_date && (
                  <span className="text-xs text-gray-400">
                    {new Date(doc.publication_date).getFullYear()}
                  </span>
                )}
              </div>
              <div className={`text-xs font-medium ${getConfidenceColor(doc.relevance_score)}`}>
                {(doc.relevance_score * 100).toFixed(0)}%
              </div>
            </div>

            <h4 className="text-sm font-medium text-gray-900 mb-2 line-clamp-2">
              {doc.title}
            </h4>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs text-gray-500">
                {doc.doi && (
                  <span className="font-mono">DOI: {doc.doi}</span>
                )}
              </div>
              {doc.url && (
                <a
                  href={doc.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-1 text-xs text-blue-600 hover:text-blue-800"
                >
                  <span>View</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Overall Confidence Indicator */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Overall Confidence</span>
          <div className="flex items-center space-x-2">
            {confidence >= 0.7 ? (
              <CheckCircle className="w-4 h-4 text-green-600" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
            )}
            <span className={`font-medium ${getConfidenceColor(confidence)}`}>
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        {confidence < 0.7 && (
          <p className="text-xs text-gray-500 mt-2">
            This response has been flagged for human review due to lower confidence.
          </p>
        )}
      </div>
    </div>
  );
};

export default SourcePanel;
