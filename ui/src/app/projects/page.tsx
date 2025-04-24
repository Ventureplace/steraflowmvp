"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '@/components/layout/Layout';
import NewProjectModal from '@/components/projects/NewProjectModal';

// Sample project data structure
interface Project {
  id: string;
  name: string;
  createdDate: string;
  status: 'active' | 'pending' | 'completed';
  quoteCount: number;
  emailConnected?: {
    provider: 'gmail' | 'outlook';
    email: string;
  };
}

export default function ProjectsPage() {
  const router = useRouter();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [projects, setProjects] = useState<Project[]>([
    {
      id: 'p1',
      name: 'Office Furniture Procurement',
      createdDate: 'Jan 15, 2023',
      status: 'active',
      quoteCount: 3,
      emailConnected: {
        provider: 'gmail',
        email: 'procurement@company.com'
      }
    },
    {
      id: 'p2',
      name: 'IT Equipment Refresh',
      createdDate: 'Mar 5, 2023',
      status: 'pending',
      quoteCount: 2
    },
    {
      id: 'p3',
      name: 'Office Supplies Q2',
      createdDate: 'Apr 1, 2023',
      status: 'active',
      quoteCount: 5,
      emailConnected: {
        provider: 'outlook',
        email: 'supplies@company.com'
      }
    }
  ]);

  const handleCreateProject = (projectData: {
    projectName: string;
    projectDescription: string;
    startDate: string;
    endDate: string;
    isEmailConnected: boolean;
    connectedEmail: string;
    emailProvider?: 'gmail' | 'outlook';
  }) => {
    // Generate a unique ID
    const id = `p${new Date().getTime()}`;
    
    // Create a new project object
    const newProject: Project = {
      id,
      name: projectData.projectName,
      createdDate: new Date().toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      }),
      status: 'active',
      quoteCount: 0,
    };
    
    // Add email connection if provided
    if (projectData.isEmailConnected && projectData.connectedEmail) {
      newProject.emailConnected = {
        provider: projectData.emailProvider || 'gmail',
        email: projectData.connectedEmail
      };
    }
    
    // Add the new project to the list
    setProjects([newProject, ...projects]);
    
    // Close the modal
    setIsModalOpen(false);
    
    // Navigate to the project dashboard
    router.push(`/projects/${id}`);
  };

  const handleViewProject = (projectId: string) => {
    router.push(`/projects/${projectId}`);
  };

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold">Projects</h1>
          <button 
            className="bg-blue-600 text-white px-4 py-2 rounded-lg"
            onClick={() => setIsModalOpen(true)}
          >
            New Project
          </button>
        </div>
        
        <div className="p-6 bg-white rounded-lg shadow">
          <div className="flex flex-col space-y-4">
            {projects.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No projects yet. Create your first project by clicking the "New Project" button.
              </div>
            ) : (
              projects.map(project => (
                <div key={project.id} className="border-b pb-4 last:border-b-0">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 
                        className="text-xl font-semibold text-blue-600 hover:text-blue-800 cursor-pointer"
                        onClick={() => handleViewProject(project.id)}
                      >
                        {project.name}
                      </h2>
                      <p className="text-gray-500">Created on {project.createdDate}</p>
                      <div className="mt-2 flex space-x-2">
                        <span 
                          className={`px-2 py-1 text-xs rounded ${
                            project.status === 'active' 
                              ? 'bg-green-100 text-green-800' 
                              : project.status === 'pending'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          {project.status.charAt(0).toUpperCase() + project.status.slice(1)}
                        </span>
                        <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                          {project.quoteCount} {project.quoteCount === 1 ? 'Quote' : 'Quotes'}
                        </span>
                        {project.emailConnected && (
                          <span className="px-2 py-1 text-xs bg-purple-100 text-purple-800 rounded">
                            {project.emailConnected.provider.charAt(0).toUpperCase() + project.emailConnected.provider.slice(1)} Connected
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      className="text-blue-600 hover:text-blue-800 text-sm"
                      onClick={() => handleViewProject(project.id)}
                    >
                      View Dashboard
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* New Project Modal */}
      {isModalOpen && (
        <NewProjectModal 
          isOpen={isModalOpen} 
          onClose={() => setIsModalOpen(false)}
          onCreateProject={handleCreateProject} 
        />
      )}
    </Layout>
  );
} 