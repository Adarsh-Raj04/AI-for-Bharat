import ReactMarkdown from 'react-markdown'

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  const isError = message.error

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-3xl ${isUser ? 'w-auto' : 'w-full'}`}>
        <div className={`rounded-lg p-4 ${
          isUser 
            ? 'bg-primary-600 text-white' 
            : isError 
              ? 'bg-red-50 border border-red-200' 
              : 'bg-white border border-gray-200'
        }`}>
          {/* Message Content */}
          <div className={`prose prose-sm max-w-none ${
            isUser ? 'prose-invert' : ''
          }`}>
            {isUser ? (
              <div className="whitespace-pre-wrap">{message.content}</div>
            ) : (
              <ReactMarkdown
                components={{
                  // Customize markdown rendering
                  table: ({node, ...props}) => (
                    <div className="overflow-x-auto my-4">
                      <table className="min-w-full divide-y divide-gray-200" {...props} />
                    </div>
                  ),
                  th: ({node, ...props}) => (
                    <th className="px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-700 uppercase" {...props} />
                  ),
                  td: ({node, ...props}) => (
                    <td className="px-3 py-2 text-sm text-gray-900 border-t" {...props} />
                  ),
                  code: ({node, inline, ...props}) => (
                    inline ? 
                      <code className="px-1 py-0.5 bg-gray-100 rounded text-sm" {...props} /> :
                      <code className="block p-2 bg-gray-100 rounded text-sm overflow-x-auto" {...props} />
                  ),
                  a: ({node, ...props}) => (
                    <a className="text-primary-600 hover:text-primary-800 underline" target="_blank" rel="noopener noreferrer" {...props} />
                  )
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>

          {/* Assistant Metadata */}
          {!isUser && !isError && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              {/* Citations */}
              {message.citations && message.citations.length > 0 && (
                <div className="mb-3">
                  <p className="text-xs font-semibold text-gray-700 mb-2">Sources:</p>
                  <div className="space-y-2">
                    {message.citations.map((citation) => (
                      <a
                        key={citation.number}
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-xs text-primary-600 hover:text-primary-800 hover:underline"
                      >
                        [{citation.number}] {citation.title}
                        {citation.source_type && (
                          <span className="text-gray-500 ml-1">
                            ({citation.source_type.toUpperCase()})
                          </span>
                        )}
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Confidence & Intent */}
              <div className="flex items-center space-x-4 text-xs text-gray-500">
                {message.confidence && (
                  <div className="flex items-center space-x-1">
                    <span>Confidence:</span>
                    <span className="font-medium">
                      {(message.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
                {message.intent && (
                  <div className="flex items-center space-x-1">
                    <span>Intent:</span>
                    <span className="font-medium capitalize">
                      {message.intent.replace('_', ' ')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
