"use client";

import { useState } from 'react';
import EmailConnectionSection from './EmailConnectionSection';

interface NewProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreateProject: (projectData: {
    projectName: string;
    projectDescription: string;
    startDate: string;
    endDate: string;
    isEmailConnected: boolean;
    connectedEmail: string;
    emailProvider?: 'gmail' | 'outlook';
  }) => void;
}

const NewProjectModal: React.FC<NewProjectModalProps> = ({ isOpen, onClose, onCreateProject }) => {
  const [selectedTab, setSelectedTab] = useState<'details' | 'email'>('details');
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [isEmailConnected, setIsEmailConnected] = useState(false);
  const [connectedEmail, setConnectedEmail] = useState('');
  const [emailProvider, setEmailProvider] = useState<'gmail' | 'outlook'>('gmail');

  const handleTabChange = (tab: 'details' | 'email') => {
    setSelectedTab(tab);
  };

  const handleSave = () => {
    // Call the onCreateProject callback with the project data
    onCreateProject({
      projectName,
      projectDescription,
      startDate,
      endDate,
      isEmailConnected,
      connectedEmail,
      emailProvider
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Create New Project</h2>
            <button 
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
              &times;
            </button>
          </div>

          {/* Tabs */}
          <div className="border-b mb-6">
            <nav className="flex space-x-8">
              <button
                className={`pb-4 font-medium text-sm ${
                  selectedTab === 'details'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => handleTabChange('details')}
              >
                Project Details
              </button>
              <button
                className={`pb-4 font-medium text-sm ${
                  selectedTab === 'email'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => handleTabChange('email')}
              >
                Email Connection
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          {selectedTab === 'details' ? (
            <div className="space-y-4">
              <div>
                <label htmlFor="project-name" className="block text-sm font-medium text-gray-700 mb-1">
                  Project Name *
                </label>
                <input
                  id="project-name"
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  required
                />
              </div>
              
              <div>
                <label htmlFor="project-description" className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  id="project-description"
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  value={projectDescription}
                  onChange={(e) => setProjectDescription(e.target.value)}
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="start-date" className="block text-sm font-medium text-gray-700 mb-1">
                    Start Date
                  </label>
                  <input
                    id="start-date"
                    type="date"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>
                
                <div>
                  <label htmlFor="end-date" className="block text-sm font-medium text-gray-700 mb-1">
                    End Date
                  </label>
                  <input
                    id="end-date"
                    type="date"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </div>
              </div>
            </div>
          ) : (
            <EmailConnectionSection 
              isEmailConnected={isEmailConnected}
              setIsEmailConnected={setIsEmailConnected}
              connectedEmail={connectedEmail}
              setConnectedEmail={setConnectedEmail}
              setEmailProvider={setEmailProvider}
            />
          )}

          <div className="mt-8 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50"
            >
              Cancel
            </button>
            {selectedTab === 'details' ? (
              <button
                type="button"
                onClick={() => handleTabChange('email')}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700"
              >
                Next: Email Connection
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSave}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700"
                disabled={!projectName}
              >
                Create Project
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewProjectModal; 