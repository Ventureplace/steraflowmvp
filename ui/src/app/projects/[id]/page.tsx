"use client";

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Layout from '@/components/layout/Layout';

// Mock project data type
interface Project {
  id: string;
  name: string;
  description?: string;
  startDate?: string;
  endDate?: string;
  createdDate: string;
  status: 'active' | 'pending' | 'completed';
  quoteCount: number;
  emailConnected?: {
    provider: 'gmail' | 'outlook';
    email: string;
  };
}

// Mock email message data type
interface EmailMessage {
  id: number;
  from: string;
  subject: string;
  date: string;
  read: boolean;
  content?: string;
}

// Mock project data - in a real app, this would come from an API or database
const mockProjects: Record<string, Project> = {
  'p1': {
    id: 'p1',
    name: 'Office Furniture Procurement',
    description: 'Procurement of new furniture for the west wing office renovation.',
    startDate: '2023-01-01',
    endDate: '2023-06-30',
    createdDate: 'Jan 15, 2023',
    status: 'active',
    quoteCount: 3,
    emailConnected: {
      provider: 'gmail',
      email: 'procurement@company.com'
    }
  },
  'p2': {
    id: 'p2',
    name: 'IT Equipment Refresh',
    description: 'Upgrading workstations and laptops for the engineering department.',
    startDate: '2023-03-01',
    endDate: '2023-07-31',
    createdDate: 'Mar 5, 2023',
    status: 'pending',
    quoteCount: 2
  },
  'p3': {
    id: 'p3',
    name: 'Office Supplies Q2',
    description: 'Regular quarterly office supplies procurement.',
    startDate: '2023-04-01',
    endDate: '2023-06-30',
    createdDate: 'Apr 1, 2023',
    status: 'active',
    quoteCount: 5,
    emailConnected: {
      provider: 'outlook',
      email: 'supplies@company.com'
    }
  }
};

// Mock email messages
const mockEmailMessages: EmailMessage[] = [
  { id: 1, from: 'supplier@example.com', subject: 'Quotation for Office Furniture', date: '2023-05-10', read: true, content: 'Dear procurement team,\n\nThank you for your inquiry. Please find attached our quotation for the requested office furniture items.\n\nPlease let me know if you have any questions.\n\nBest regards,\nSupplier Team' },
  { id: 2, from: 'vendor@example.com', subject: 'Updated pricing for IT equipment', date: '2023-05-09', read: false, content: 'Hello,\n\nWe have updated our pricing for the IT equipment you requested. The new quote reflects a 5% discount on all items.\n\nRegards,\nVendor Sales Team' },
  { id: 3, from: 'sales@furnishings.com', subject: 'Special Discount Offer', date: '2023-05-08', read: true, content: 'SPECIAL OFFER:\n\nWe are offering a 10% discount on all orders placed before the end of the month.\n\nFurnishings Team' },
  { id: 4, from: 'support@techsupplier.com', subject: 'Your Recent Order', date: '2023-05-07', read: true, content: 'Thank you for your order!\n\nYour recent order #12345 has been processed and will be shipped within 3 business days.\n\nTech Supplier Support' },
  { id: 5, from: 'info@officesupplies.com', subject: 'Catalog Update', date: '2023-05-06', read: false, content: 'Our new catalog is now available.\n\nCheck out our latest products and special offers for this month.\n\nOffice Supplies Team' }
];

// Mock parsed quotes data for the comparison view
interface ParsedQuoteItem {
  id: string;
  category: string;
  name: string;
  description?: string;
  prices: {
    [supplier: string]: {
      price: number;
      quantity: number;
      notes?: string;
    }
  };
}

const mockParsedQuoteItems: ParsedQuoteItem[] = [
  {
    id: 'item1',
    category: 'Desks',
    name: 'Executive Desk',
    description: 'Large executive desk with drawers',
    prices: {
      'Office Solutions Inc.': { price: 1200.00, quantity: 5, notes: 'Includes delivery' },
      'Furniture Express': { price: 1150.00, quantity: 5, notes: 'Additional assembly fee' },
      'Modern Office Supplies': { price: 1300.00, quantity: 5, notes: '5-year warranty' },
    }
  },
  {
    id: 'item2',
    category: 'Chairs',
    name: 'Ergonomic Chair',
    description: 'Adjustable office chair with lumbar support',
    prices: {
      'Office Solutions Inc.': { price: 450.00, quantity: 10 },
      'Furniture Express': { price: 425.00, quantity: 10 },
      'Modern Office Supplies': { price: 475.00, quantity: 10, notes: 'Premium fabric' },
    }
  },
  {
    id: 'item3',
    category: 'Storage',
    name: 'Filing Cabinet',
    description: '4-drawer filing cabinet with lock',
    prices: {
      'Office Solutions Inc.': { price: 350.00, quantity: 8 },
      'Furniture Express': { price: 375.00, quantity: 8, notes: 'Fire resistant' },
      'Modern Office Supplies': { price: 325.00, quantity: 8 },
    }
  },
  {
    id: 'item4',
    category: 'Accessories',
    name: 'Monitor Stand',
    description: 'Adjustable height monitor stand',
    prices: {
      'Office Solutions Inc.': { price: 75.00, quantity: 20 },
      'Furniture Express': { price: 85.00, quantity: 20, notes: 'Includes cable management' },
      'Modern Office Supplies': { price: 65.00, quantity: 20 },
    }
  },
];

export default function ProjectDashboardPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;
  
  const [project, setProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'emails' | 'quotes' | 'parsed_quotes'>('overview');
  const [emailMessages, setEmailMessages] = useState<EmailMessage[]>([]);
  const [selectedEmail, setSelectedEmail] = useState<EmailMessage | null>(null);

  useEffect(() => {
    // Simulate fetching project data
    const fetchProject = async () => {
      setIsLoading(true);
      try {
        // In a real app, this would be an API call
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Check if the project exists in our mock data
        if (mockProjects[projectId]) {
          setProject(mockProjects[projectId]);
          
          // If project has email connected, also load emails
          if (mockProjects[projectId].emailConnected) {
            setEmailMessages(mockEmailMessages);
          }
        }
      } catch (error) {
        console.error('Error fetching project:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchProject();
  }, [projectId]);

  const handleBackToProjects = () => {
    router.push('/projects');
  };

  const handleEmailClick = (email: EmailMessage) => {
    setSelectedEmail(email);
  };

  const handleCloseEmail = () => {
    setSelectedEmail(null);
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center min-h-screen -mt-16">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      </Layout>
    );
  }

  if (!project) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center space-x-2">
            <button
              onClick={handleBackToProjects}
              className="text-blue-600 hover:text-blue-800"
            >
              ← Back to Projects
            </button>
          </div>
          
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Project Not Found</h2>
            <p className="text-gray-600 mb-6">The project you're looking for doesn't exist or has been removed.</p>
            <button
              onClick={handleBackToProjects}
              className="px-4 py-2 bg-blue-600 text-white rounded-md"
            >
              Return to Projects
            </button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center space-x-2">
          <button
            onClick={handleBackToProjects}
            className="text-blue-600 hover:text-blue-800"
          >
            ← Back to Projects
          </button>
        </div>
        
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold">{project.name}</h1>
            <div className="flex space-x-3 mt-2">
              <span className={`px-2 py-1 text-xs rounded ${
                project.status === 'active' 
                  ? 'bg-green-100 text-green-800' 
                  : project.status === 'pending'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-gray-100 text-gray-800'
              }`}>
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
            <p className="text-gray-500 mt-1">Created on {project.createdDate}</p>
          </div>
          
          <div className="space-x-3">
            <button className="px-4 py-2 bg-white text-blue-600 border border-blue-600 rounded-md">
              Edit Project
            </button>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-md">
              Add Quote
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b">
          <nav className="flex space-x-8">
            <button
              className={`pb-4 font-medium text-sm ${
                activeTab === 'overview'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button
              className={`pb-4 font-medium text-sm ${
                activeTab === 'emails'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('emails')}
            >
              Email Messages
              {project.emailConnected ? 
                <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full text-xs">
                  {emailMessages.filter(m => !m.read).length}
                </span> : null
              }
            </button>
            <button
              className={`pb-4 font-medium text-sm ${
                activeTab === 'quotes'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('quotes')}
            >
              Quotes
              <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full text-xs">
                {project.quoteCount}
              </span>
            </button>
            <button
              className={`pb-4 font-medium text-sm ${
                activeTab === 'parsed_quotes'
                  ? 'border-b-2 border-blue-500 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('parsed_quotes')}
            >
              Parsed Quotes
              <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full text-xs">
                {project.quoteCount}
              </span>
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Project Details */}
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Project Details</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-500">Description</p>
                    <p className="text-gray-800">{project.description || 'No description provided.'}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Start Date</p>
                      <p className="text-gray-800">{project.startDate || 'Not specified'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">End Date</p>
                      <p className="text-gray-800">{project.endDate || 'Not specified'}</p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold mb-2">Quote Summary</h3>
                  <div className="space-y-1">
                    <p className="text-3xl font-bold text-blue-600">{project.quoteCount}</p>
                    <p className="text-gray-500">Total Quotes</p>
                  </div>
                  <button 
                    className="mt-4 text-blue-600 text-sm hover:text-blue-800"
                    onClick={() => setActiveTab('quotes')}
                  >
                    View All Quotes →
                  </button>
                </div>
                
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold mb-2">Email Activity</h3>
                  {project.emailConnected ? (
                    <div>
                      <div className="space-y-1">
                        <p className="text-3xl font-bold text-blue-600">{emailMessages.filter(m => !m.read).length}</p>
                        <p className="text-gray-500">Unread Messages</p>
                      </div>
                      <button 
                        className="mt-4 text-blue-600 text-sm hover:text-blue-800"
                        onClick={() => setActiveTab('emails')}
                      >
                        View All Messages →
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <p className="text-gray-500">No email connected</p>
                      <button className="mt-4 text-blue-600 text-sm hover:text-blue-800">
                        Connect Email →
                      </button>
                    </div>
                  )}
                </div>
                
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold mb-2">Project Timeline</h3>
                  {project.startDate && project.endDate ? (
                    <div className="space-y-1">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '45%' }}></div>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>{project.startDate}</span>
                        <span>{project.endDate}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-500">Timeline not configured</p>
                  )}
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'emails' && (
            <div className="bg-white rounded-lg shadow">
              {!project.emailConnected ? (
                <div className="p-8 text-center">
                  <h2 className="text-xl font-semibold mb-2">No Email Connected</h2>
                  <p className="text-gray-500 mb-4">Connect an email account to view messages related to this project.</p>
                  <button className="px-4 py-2 bg-blue-600 text-white rounded-md">
                    Connect Email
                  </button>
                </div>
              ) : selectedEmail ? (
                <div className="p-6">
                  <div className="flex justify-between items-center mb-4">
                    <button
                      className="text-blue-600 hover:text-blue-800"
                      onClick={handleCloseEmail}
                    >
                      ← Back to messages
                    </button>
                    <div className="text-gray-500 text-sm">{selectedEmail.date}</div>
                  </div>
                  
                  <div className="border-b pb-4 mb-4">
                    <h2 className="text-xl font-semibold">{selectedEmail.subject}</h2>
                    <p className="text-gray-500">From: {selectedEmail.from}</p>
                  </div>
                  
                  <div className="prose max-w-none whitespace-pre-line">
                    {selectedEmail.content}
                  </div>
                  
                  <div className="mt-6 flex justify-end space-x-3">
                    <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50">
                      Reply
                    </button>
                    <button className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700">
                      Forward
                    </button>
                  </div>
                </div>
              ) : (
                <div className="divide-y">
                  {emailMessages.map((message) => (
                    <div 
                      key={message.id}
                      className={`p-4 cursor-pointer hover:bg-gray-50 ${!message.read ? 'bg-blue-50' : ''}`}
                      onClick={() => handleEmailClick(message)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-grow">
                          <div className="flex items-center">
                            {!message.read && (
                              <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                            )}
                            <h3 className={`text-base ${!message.read ? 'font-semibold' : ''}`}>
                              {message.subject}
                            </h3>
                          </div>
                          <p className="text-sm text-gray-500">From: {message.from}</p>
                        </div>
                        <div className="text-xs text-gray-500">{message.date}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'quotes' && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold">Quotes</h2>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm">
                  Add Quote
                </button>
              </div>
              
              {project.quoteCount === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No quotes yet. Click "Add Quote" to create your first quote.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Quote #
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Supplier
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Amount
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">Q-2023-001</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Office Solutions Inc.</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">May 2, 2023</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$12,450.00</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded">Approved</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900 mr-3">View</button>
                          <button className="text-blue-600 hover:text-blue-900">Edit</button>
                        </td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">Q-2023-002</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Furniture Express</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">May 5, 2023</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$11,875.00</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded">Pending</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900 mr-3">View</button>
                          <button className="text-blue-600 hover:text-blue-900">Edit</button>
                        </td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">Q-2023-003</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Modern Office Supplies</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">May 8, 2023</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">$13,200.00</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded">Pending</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900 mr-3">View</button>
                          <button className="text-blue-600 hover:text-blue-900">Edit</button>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'parsed_quotes' && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold">Parsed Quotes Comparison</h2>
                <div className="flex space-x-2">
                  <button className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-md text-sm">
                    Export to Excel
                  </button>
                  <button className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm">
                    Re-parse Quotes
                  </button>
                </div>
              </div>
              
              {project.quoteCount === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No quotes to parse. Add quotes first to enable AI-powered parsing and comparison.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <div className="border rounded-lg mb-4 bg-blue-50 p-4 text-sm text-blue-800">
                    <p>SteraFlow AI has analyzed and standardized the quotes from 3 different suppliers. Items have been categorized and matched across quotes for easy comparison.</p>
                  </div>
                  
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Item
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Category
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Office Solutions Inc.
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Furniture Express
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Modern Office Supplies
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Best Price
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {mockParsedQuoteItems.map((item) => {
                        // Find the supplier with the best price
                        const suppliers = Object.keys(item.prices);
                        const bestSupplier = suppliers.reduce((best, current) => 
                          (item.prices[current].price < item.prices[best].price) ? current : best, 
                          suppliers[0]
                        );
                        
                        return (
                          <tr key={item.id}>
                            <td className="px-6 py-4">
                              <div className="text-sm font-medium text-gray-900">{item.name}</div>
                              <div className="text-sm text-gray-500">{item.description}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {item.category}
                            </td>
                            {suppliers.map(supplier => (
                              <td key={`${item.id}-${supplier}`} className="px-6 py-4">
                                <div className={`text-sm font-medium ${supplier === bestSupplier ? 'text-green-600' : 'text-gray-900'}`}>
                                  ${item.prices[supplier].price.toFixed(2)}
                                </div>
                                <div className="text-xs text-gray-500">
                                  Qty: {item.prices[supplier].quantity}
                                  {item.prices[supplier].notes && (
                                    <span className="block">{item.prices[supplier].notes}</span>
                                  )}
                                </div>
                              </td>
                            ))}
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                                {bestSupplier}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                      <tr className="bg-gray-50 font-medium">
                        <td className="px-6 py-4 text-sm text-gray-900" colSpan={2}>Total (all items)</td>
                        <td className="px-6 py-4 text-sm text-gray-900">$12,450.00</td>
                        <td className="px-6 py-4 text-sm text-gray-900">$11,875.00</td>
                        <td className="px-6 py-4 text-sm text-gray-900">$13,200.00</td>
                        <td className="px-6 py-4">
                          <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                            Furniture Express
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  
                  <div className="mt-6 flex justify-between">
                    <div className="text-sm text-gray-500">
                      <p>* Prices include all quoted items with specified quantities</p>
                      <p>* Items have been matched based on specifications and features</p>
                    </div>
                    <button className="px-4 py-2 bg-green-600 text-white rounded-md text-sm">
                      Generate Procurement Report
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
} 