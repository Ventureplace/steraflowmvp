import Layout from '@/components/layout/Layout';

export default function CompareQuotesPage() {
  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Compare Quotes</h1>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Project to Compare Quotes
            </label>
            <div className="flex space-x-4">
              <select
                className="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="" disabled selected>Select a project</option>
                <option value="project1">Office Furniture Procurement</option>
                <option value="project2">IT Equipment Refresh</option>
                <option value="project3">Office Supplies Q2</option>
              </select>
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700"
              >
                Load Quotes
              </button>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-40">
                    Item / Supplier
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Office Solutions Inc.
                    <div className="text-xs font-normal mt-1">Quote #OS-2023-156</div>
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Furniture Express
                    <div className="text-xs font-normal mt-1">Quote #FE-4567</div>
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Modern Office Supplies
                    <div className="text-xs font-normal mt-1">Quote #MO-78905</div>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap font-medium text-sm text-gray-900">
                    Executive Desk
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$1,200.00</div>
                    <div className="text-xs text-gray-500">Qty: 5</div>
                    <div className="text-xs text-gray-500">Total: $6,000.00</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center bg-green-50">
                    <div className="font-medium text-green-700">$1,050.00</div>
                    <div className="text-xs text-gray-500">Qty: 5</div>
                    <div className="text-xs text-gray-500">Total: $5,250.00</div>
                    <div className="mt-1 text-xs text-green-700 font-medium">BEST PRICE</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$1,350.00</div>
                    <div className="text-xs text-gray-500">Qty: 5</div>
                    <div className="text-xs text-gray-500">Total: $6,750.00</div>
                  </td>
                </tr>
                
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap font-medium text-sm text-gray-900">
                    Office Chair (Ergonomic)
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center bg-green-50">
                    <div className="font-medium text-green-700">$250.00</div>
                    <div className="text-xs text-gray-500">Qty: 10</div>
                    <div className="text-xs text-gray-500">Total: $2,500.00</div>
                    <div className="mt-1 text-xs text-green-700 font-medium">BEST PRICE</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$275.00</div>
                    <div className="text-xs text-gray-500">Qty: 10</div>
                    <div className="text-xs text-gray-500">Total: $2,750.00</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$290.00</div>
                    <div className="text-xs text-gray-500">Qty: 10</div>
                    <div className="text-xs text-gray-500">Total: $2,900.00</div>
                  </td>
                </tr>
                
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap font-medium text-sm text-gray-900">
                    Filing Cabinet
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$180.00</div>
                    <div className="text-xs text-gray-500">Qty: 8</div>
                    <div className="text-xs text-gray-500">Total: $1,440.00</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                    <div className="font-medium">$195.00</div>
                    <div className="text-xs text-gray-500">Qty: 8</div>
                    <div className="text-xs text-gray-500">Total: $1,560.00</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-center bg-green-50">
                    <div className="font-medium text-green-700">$175.00</div>
                    <div className="text-xs text-gray-500">Qty: 8</div>
                    <div className="text-xs text-gray-500">Total: $1,400.00</div>
                    <div className="mt-1 text-xs text-green-700 font-medium">BEST PRICE</div>
                  </td>
                </tr>
              </tbody>
              <tfoot>
                <tr className="bg-gray-50">
                  <td className="px-4 py-3 whitespace-nowrap font-bold text-sm text-gray-900">
                    Total Quote Amount
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-center font-bold text-sm">
                    $9,940.00
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-center font-bold text-sm">
                    $9,560.00
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-center font-bold text-green-700 text-sm bg-green-50">
                    $9,050.00
                    <div className="mt-1 text-xs text-green-700 font-medium">LOWEST OVERALL</div>
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
          
          <div className="mt-6 flex justify-between">
            <button
              type="button"
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50"
            >
              Export Comparison
            </button>
            <button
              type="button"
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700"
            >
              Create Purchase Order
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
} 