"use client";

import { useState } from 'react';

interface EmailConnectionSectionProps {
  isEmailConnected: boolean;
  setIsEmailConnected: (value: boolean) => void;
  connectedEmail: string;
  setConnectedEmail: (email: string) => void;
  setEmailProvider?: (provider: 'gmail' | 'outlook') => void;
}

const EmailConnectionSection: React.FC<EmailConnectionSectionProps> = ({
  isEmailConnected,
  setIsEmailConnected,
  connectedEmail,
  setConnectedEmail,
  setEmailProvider
}) => {
  const [emailType, setEmailType] = useState<'gmail' | 'outlook'>('gmail');
  const [emailAddress, setEmailAddress] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [emailMessages, setEmailMessages] = useState<any[]>([]);
  const [showMessages, setShowMessages] = useState(false);

  const handleEmailTypeChange = (type: 'gmail' | 'outlook') => {
    setEmailType(type);
    // Update parent component if callback is provided
    if (setEmailProvider) {
      setEmailProvider(type);
    }
  };

  const handleConnect = async () => {
    if (!emailAddress || !password) {
      setError('Please enter both email and password');
      return;
    }

    setError('');
    setIsLoading(true);

    try {
      // This would be an actual API call in a real application
      // For now, we're simulating a successful connection
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setIsEmailConnected(true);
      setConnectedEmail(emailAddress);
      
      // Update parent component with email provider if callback is provided
      if (setEmailProvider) {
        setEmailProvider(emailType);
      }
      
      // Simulate fetching email messages
      const mockMessages = [
        { id: 1, from: 'supplier@example.com', subject: 'Quotation for Office Furniture', date: '2023-05-10', read: true },
        { id: 2, from: 'vendor@example.com', subject: 'Updated pricing for IT equipment', date: '2023-05-09', read: false },
        { id: 3, from: 'sales@furnishings.com', subject: 'Special Discount Offer', date: '2023-05-08', read: true },
        { id: 4, from: 'support@techsupplier.com', subject: 'Your Recent Order', date: '2023-05-07', read: true },
        { id: 5, from: 'info@officesupplies.com', subject: 'Catalog Update', date: '2023-05-06', read: false }
      ];
      
      setEmailMessages(mockMessages);
    } catch (err) {
      setError('Failed to connect. Please check your credentials and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisconnect = () => {
    setIsEmailConnected(false);
    setConnectedEmail('');
    setEmailMessages([]);
    setShowMessages(false);
  };

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-medium">Connect Email Account</h3>
      <p className="text-gray-500">
        Connect an email account to automatically track communications related to this project.
      </p>

      {!isEmailConnected ? (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email Provider
            </label>
            <div className="flex space-x-4">
              <button
                type="button"
                className={`px-4 py-2 rounded-md ${
                  emailType === 'gmail'
                    ? 'bg-blue-50 border border-blue-500 text-blue-700'
                    : 'bg-gray-50 border border-gray-300 text-gray-700'
                }`}
                onClick={() => handleEmailTypeChange('gmail')}
              >
                Gmail
              </button>
              <button
                type="button"
                className={`px-4 py-2 rounded-md ${
                  emailType === 'outlook'
                    ? 'bg-blue-50 border border-blue-500 text-blue-700'
                    : 'bg-gray-50 border border-gray-300 text-gray-700'
                }`}
                onClick={() => handleEmailTypeChange('outlook')}
              >
                Outlook
              </button>
            </div>
          </div>

          <div>
            <label htmlFor="email-address" className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <input
              id="email-address"
              type="email"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              value={emailAddress}
              onChange={(e) => setEmailAddress(e.target.value)}
              placeholder={emailType === 'gmail' ? 'username@gmail.com' : 'username@outlook.com'}
            />
          </div>

          <div>
            <label htmlFor="email-password" className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              id="email-password"
              type="password"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Your password"
            />
          </div>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <div className="flex justify-end">
            <button
              type="button"
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:opacity-50"
              onClick={handleConnect}
              disabled={isLoading || !emailAddress || !password}
            >
              {isLoading ? 'Connecting...' : 'Connect Email'}
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-md">
            <div className="flex items-center">
              <div className="flex-grow">
                <p className="text-green-700 font-medium">
                  ✓ Connected to {emailType === 'gmail' ? 'Gmail' : 'Outlook'}
                </p>
                <p className="text-sm text-gray-600">{connectedEmail}</p>
              </div>
              <button
                type="button"
                className="text-red-600 text-sm hover:text-red-800"
                onClick={handleDisconnect}
              >
                Disconnect
              </button>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-medium">Recent Messages</h4>
              <button
                type="button"
                className="text-blue-600 text-sm hover:text-blue-800"
                onClick={() => setShowMessages(!showMessages)}
              >
                {showMessages ? 'Hide Messages' : 'Show Messages'}
              </button>
            </div>

            {showMessages && (
              <div className="border border-gray-200 rounded-md overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        From
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Subject
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Date
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {emailMessages.map((message) => (
                      <tr key={message.id} className={message.read ? '' : 'font-semibold bg-blue-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {message.from}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {message.subject}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {message.date}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmailConnectionSection; 