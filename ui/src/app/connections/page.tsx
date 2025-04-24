"use client";

import Layout from '@/components/layout/Layout';
import { useState, useEffect } from 'react';

export default function ConnectionsPage() {
  // State to track connection status
  const [connections, setConnections] = useState({
    gmail: false,
    bigQuery: false
  });

  // Simulate checking connection status with backend
  useEffect(() => {
    // This would be replaced with actual API calls to check connection status
    const checkConnectionStatus = async () => {
      try {
        // Simulate API response delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // This would be replaced with actual backend responses
        // For now, all connections default to disconnected
        setConnections({
          gmail: false,
          bigQuery: false
        });
      } catch (error) {
        console.error("Error checking connection status:", error);
      }
    };

    checkConnectionStatus();
  }, []);

  // Function to handle connection attempt
  const handleConnect = (connectionName: 'gmail' | 'bigQuery') => {
    // This would trigger an auth flow or API call in production
    // For demo purposes, just toggle the connection status
    setConnections(prev => ({
      ...prev,
      [connectionName]: true
    }));
  };

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold">Connections</h1>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg">
            Add Connection
          </button>
        </div>
        
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-4">Email Connections</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-6 bg-white rounded-lg shadow">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Gmail</h3>
                <span className={`px-2 py-1 text-xs rounded ${
                  connections.gmail 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {connections.gmail ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <p className="text-gray-500 mt-2">
                {connections.gmail ? 'procurement@company.com' : 'Not configured'}
              </p>
              <div className="mt-4 flex space-x-3">
                {connections.gmail ? (
                  <>
                    <button className="text-sm text-blue-600 hover:underline">View Messages</button>
                    <button className="text-sm text-blue-600 hover:underline">Configure</button>
                  </>
                ) : (
                  <>
                    <button 
                      className="text-sm text-blue-600 hover:underline"
                      onClick={() => handleConnect('gmail')}
                    >
                      Connect
                    </button>
                    <button className="text-sm text-blue-600 hover:underline">Learn More</button>
                  </>
                )}
              </div>
            </div>
            
            <div className="p-6 bg-white rounded-lg shadow border-2 border-dashed border-gray-300 flex items-center justify-center">
              <button className="text-gray-500 hover:text-blue-600 flex flex-col items-center">
                <span className="text-xl mb-2">+</span>
                <span>Add Email Connection</span>
              </button>
            </div>
          </div>
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-4">Data Warehouse</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 bg-white rounded-lg shadow">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">BigQuery</h3>
                <span className={`px-2 py-1 text-xs rounded ${
                  connections.bigQuery 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {connections.bigQuery ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <p className="text-gray-500 mt-2">
                {connections.bigQuery ? 'procurement-analytics-123456' : 'Not connected'}
              </p>
              <div className="mt-4 flex space-x-3">
                {connections.bigQuery ? (
                  <>
                    <button className="text-sm text-blue-600 hover:underline">View Data</button>
                    <button className="text-sm text-blue-600 hover:underline">Manage</button>
                  </>
                ) : (
                  <>
                    <button 
                      className="text-sm text-blue-600 hover:underline"
                      onClick={() => handleConnect('bigQuery')}
                    >
                      Connect
                    </button>
                    <button className="text-sm text-blue-600 hover:underline">Learn More</button>
                  </>
                )}
              </div>
            </div>
            
            <div className="p-6 bg-white rounded-lg shadow border-2 border-dashed border-gray-300 flex items-center justify-center">
              <button className="text-gray-500 hover:text-blue-600 flex flex-col items-center">
                <span className="text-xl mb-2">+</span>
                <span>Add Data Warehouse</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
} 