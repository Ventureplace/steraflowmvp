"use client";

import Layout from '@/components/layout/Layout';
import { useState, useEffect } from 'react';
import { MicrophoneIcon, PaperAirplaneIcon, PlusIcon, ChevronDownIcon, XMarkIcon, ClockIcon } from '@heroicons/react/24/outline';

export default function ChatPage() {
  const [messages, setMessages] = useState<Array<{ text: string; isUser: boolean; timestamp?: Date }>>([]);
  const [input, setInput] = useState("");
  const [showProjectList, setShowProjectList] = useState(false);
  const [showChatHistory, setShowChatHistory] = useState(false);
  const [selectedProject, setSelectedProject] = useState<{ id: number; name: string; status: string } | null>(null);
  
  // Sample projects - you would fetch these from your API
  const projects = [
    { id: 1, name: "Office Equipment Procurement", status: "Active" },
    { id: 2, name: "IT Hardware Refresh", status: "Active" },
    { id: 3, name: "Manufacturing Supplies", status: "Pending" },
    { id: 4, name: "Catering Services", status: "Completed" },
    { id: 5, name: "Construction Materials", status: "Active" }
  ];

  // Sample chat history - would be fetched from backend in real app
  const chatHistory = [
    { 
      id: 1, 
      title: "Office Equipment Pricing", 
      preview: "I need help with office equipment pricing...",
      date: new Date(2023, 7, 15) 
    },
    { 
      id: 2, 
      title: "Supplier Evaluation", 
      preview: "How can I evaluate new suppliers for our project?",
      date: new Date(2023, 7, 10) 
    },
    { 
      id: 3, 
      title: "Procurement Process", 
      preview: "What's the typical procurement process for IT hardware?",
      date: new Date(2023, 7, 5) 
    },
  ];

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      setMessages([...messages, { text: input, isUser: true, timestamp: new Date() }]);
      setInput("");
      // Simulate response - in a real app, you'd call your API here
      setTimeout(() => {
        setMessages(prev => [...prev, { 
          text: "I'm your procurement assistant. How can I help you today?", 
          isUser: false,
          timestamp: new Date()
        }]);
      }, 1000);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };
  
  const toggleProjectList = () => {
    setShowProjectList(!showProjectList);
    if (showProjectList) setShowChatHistory(false);
  };
  
  const toggleChatHistory = () => {
    setShowChatHistory(!showChatHistory);
    if (showChatHistory) setShowProjectList(false);
  };
  
  const selectProject = (project: typeof projects[0]) => {
    setSelectedProject(project);
    setInput(`Tell me about the ${project.name} project`);
    setShowProjectList(false);
  };
  
  const clearSelectedProject = () => {
    setSelectedProject(null);
  };
  
  const loadChatHistory = (chatId: number) => {
    // In a real app, you would fetch the chat history from your backend
    const selectedChat = chatHistory.find(chat => chat.id === chatId);
    if (selectedChat) {
      setMessages([
        { text: selectedChat.preview, isUser: true, timestamp: selectedChat.date },
        { text: "Here's some information I found about that...", isUser: false, timestamp: selectedChat.date }
      ]);
      setShowChatHistory(false);
    }
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <Layout>
      <div className="flex flex-col h-full bg-white relative">
        {/* Header with model selector and temporary button */}
        <div className="absolute top-2 right-8 flex items-center space-x-2 z-10">
          <div className="flex items-center">
            <button 
              className="bg-transparent border border-gray-300 text-sm font-medium rounded-md py-1 px-3 flex items-center space-x-1 hover:bg-gray-50"
              onClick={toggleChatHistory}
            >
              <span>Chat History</span>
              <ChevronDownIcon className="h-4 w-4" />
            </button>
            
            {/* Chat History Dropdown */}
            {showChatHistory && (
              <div className="absolute top-full right-0 mt-1 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-20">
                {chatHistory.length > 0 ? (
                  <div className="max-h-72 overflow-y-auto">
                    {chatHistory.map(chat => (
                      <button
                        key={chat.id}
                        onClick={() => loadChatHistory(chat.id)}
                        className="w-full text-left p-3 hover:bg-gray-50 border-b border-gray-100 flex flex-col"
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-medium text-gray-800">{chat.title}</span>
                          <span className="text-xs text-gray-500">{formatDate(chat.date)}</span>
                        </div>
                        <span className="text-xs text-gray-500 truncate">{chat.preview}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 text-center text-gray-500">
                    No chat history available
                  </div>
                )}
              </div>
            )}
          </div>
          <button className="flex items-center space-x-1 border border-gray-300 rounded-md py-1 px-3 text-sm hover:bg-gray-50">
            <span className="inline-block w-2 h-2 rounded-full bg-gray-400"></span>
            <span>Temporary</span>
          </button>
        </div>
        
        {/* Main chat area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center">
              <h1 className="text-4xl font-semibold mb-6 text-gray-800">What can I help with?</h1>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <div 
                  key={index} 
                  className={`px-4 py-6 ${message.isUser ? 'bg-white' : 'bg-gray-50'}`}
                >
                  <div className="max-w-3xl mx-auto flex">
                    <div className="w-8 h-8 rounded-full flex-shrink-0 mr-4 flex items-center justify-center">
                      {message.isUser ? (
                        <div className="w-full h-full bg-gray-300 rounded-full flex items-center justify-center text-white">
                          U
                        </div>
                      ) : (
                        <div className="w-full h-full bg-green-600 rounded-full flex items-center justify-center text-white">
                          AI
                        </div>
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="mb-1">
                        {message.text}
                      </div>
                      {message.timestamp && (
                        <div className="text-xs text-gray-400 mt-1 flex items-center">
                          <ClockIcon className="h-3 w-3 mr-1" />
                          {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="border-t border-gray-200 mt-auto py-4 px-4 sm:px-6 md:px-8">
          <div className="max-w-3xl mx-auto">
            {/* Context indicator - only shown when a project is selected */}
            {selectedProject && (
              <div className="mb-3 flex items-center justify-center">
                <div className="bg-blue-50 border border-blue-200 rounded-full px-3 py-1 flex items-center text-sm">
                  <span className="text-blue-700 font-medium mr-1">Context:</span>
                  <span className="text-blue-600">{selectedProject.name}</span>
                  <button 
                    className="ml-2 text-blue-400 hover:text-blue-600"
                    onClick={clearSelectedProject}
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
            
            <form onSubmit={handleSendMessage} className="relative flex items-center">
              <div className="relative">
                <button 
                  type="button"
                  onClick={toggleProjectList}
                  className={`absolute left-3 top-1/2 transform -translate-y-1/2 p-1.5 rounded-full 
                    ${selectedProject ? 'bg-blue-100 text-blue-600' : 'hover:bg-gray-100 text-gray-500'}`}
                  aria-label="Select project"
                >
                  <PlusIcon className="h-5 w-5" />
                </button>
                
                {/* Project list popup */}
                {showProjectList && (
                  <div className="absolute bottom-full left-0 mb-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                    <div className="p-3 border-b border-gray-200">
                      <h3 className="font-medium text-gray-900">Available Projects</h3>
                      <p className="text-xs text-gray-500 mt-1">Select a project to discuss</p>
                    </div>
                    <div className="max-h-60 overflow-y-auto p-2">
                      {projects.map(project => (
                        <button
                          key={project.id}
                          onClick={() => selectProject(project)}
                          className={`w-full text-left p-2 rounded-md flex items-center justify-between group
                            ${selectedProject?.id === project.id ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
                        >
                          <span className={`font-medium ${selectedProject?.id === project.id ? 'text-blue-700' : 'text-gray-700'}`}>
                            {project.name}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            project.status === 'Active' ? 'bg-green-100 text-green-800' :
                            project.status === 'Pending' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {project.status}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything"
                className="w-full py-3 pl-12 pr-14 border border-gray-300 rounded-full focus:outline-none focus:ring-1 focus:border-gray-300 focus:ring-gray-300 shadow-sm"
              />
              
              <button 
                type="submit" 
                className="absolute right-10 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 disabled:opacity-50"
                disabled={!input.trim()}
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>

              <button 
                type="button" 
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                aria-label="Voice input"
              >
                <MicrophoneIcon className="h-5 w-5" />
              </button>
            </form>
          </div>
          <div className="max-w-3xl mx-auto mt-2 text-xs text-center text-gray-500">
            SteraFlow Procurement Assistant can make mistakes. Consider checking important information.
          </div>
        </div>
      </div>
    </Layout>
  );
} 