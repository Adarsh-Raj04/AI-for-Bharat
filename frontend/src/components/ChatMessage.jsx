import ReactMarkdown from 'react-markdown'

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  const isError = message.error

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`${isUser ? 'max-w-[85%] sm:max-w-2xl' : 'w-full max-w-full sm:max-w-4xl'}`}>
        <div className={`rounded-lg p-3 sm:p-4 ${
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
              <div className="whitespace-pre-wrap text-sm sm:text-base">{message.content}</div>
            ) : (
              <ReactMarkdown
                components={{
                  // Customize markdown rendering
                  table: ({node, ...props}) => (
                    <div className="overflow-x-auto my-3 sm:my-4 -mx-3 sm:mx-0 px-3 sm:px-0">
                      <table className="min-w-full divide-y divide-gray-200 text-xs sm:text-sm" {...props} />
                    </div>
                  ),
                  th: ({node, ...props}) => (
                    <th className="px-2 sm:px-3 py-1.5 sm:py-2 bg-gray-50 text-left text-xs font-medium text-gray-700 uppercase" {...props} />
                  ),
                  td: ({node, ...props}) => (
                    <td className="px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-gray-900 border-t" {...props} />
                  ),
                  code: ({node, inline, ...props}) => (
                    inline ? 
                      <code className="px-1 py-0.5 bg-gray-100 rounded text-xs sm:text-sm" {...props} /> :
                      <code className="block p-2 bg-gray-100 rounded text-xs sm:text-sm overflow-x-auto" {...props} />
                  ),
                  a: ({node, ...props}) => (
                    <a className="text-primary-600 hover:text-primary-800 underline break-words" target="_blank" rel="noopener noreferrer" {...props} />
                  ),
                  p: ({node, ...props}) => (
                    <p className="text-sm sm:text-base mb-2 sm:mb-3" {...props} />
                  ),
                  ul: ({node, ...props}) => (
                    <ul className="text-sm sm:text-base list-disc pl-4 sm:pl-5 mb-2 sm:mb-3" {...props} />
                  ),
                  ol: ({node, ...props}) => (
                    <ol className="text-sm sm:text-base list-decimal pl-4 sm:pl-5 mb-2 sm:mb-3" {...props} />
                  ),
                  h1: ({node, ...props}) => (
                    <h1 className="text-lg sm:text-xl font-bold mb-2 sm:mb-3" {...props} />
                  ),
                  h2: ({node, ...props}) => (
                    <h2 className="text-base sm:text-lg font-bold mb-2 sm:mb-3" {...props} />
                  ),
                  h3: ({node, ...props}) => (
                    <h3 className="text-sm sm:text-base font-bold mb-1 sm:mb-2" {...props} />
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>

          {/* Assistant Metadata */}
          {!isUser && !isError && (
            <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-gray-200">
              {/* Citations */}
              {message.citations && message.citations.length > 0 && (
                <div className="mb-2 sm:mb-3">
                  <p className="text-xs font-semibold text-gray-700 mb-1.5 sm:mb-2">Sources:</p>
                  <div className="space-y-1.5 sm:space-y-2">
                    {message.citations.map((citation) => (
                      <a
                        key={citation.number}
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-xs text-primary-600 hover:text-primary-800 hover:underline break-words"
                      >
                        <span className="font-medium">[{citation.number}]</span> {citation.title}
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
              <div className="flex flex-wrap items-center gap-x-3 sm:gap-x-4 gap-y-1 text-xs text-gray-500">
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
