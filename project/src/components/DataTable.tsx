import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

const data = [
  {
    commodity: 'Tomato',
    market: 'Madanapalle',
    region: 'Madanapalle',
    currentPrice: 2300,
    predictedPrice: 2400,
    change: 4.3
  },
  {
    commodity: 'Rice',
    market: 'East Godavari',
    region: 'East Godavari',
    currentPrice: 3200,
    predictedPrice: 3300,
    change: 3.125
  },
  // Add more sample data as needed
];

const DataTable = () => {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Commodity
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Market
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Region
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Current Price
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Predicted Price
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Change (%)
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {data.map((row, index) => (
            <tr key={index}>
              <td className="px-6 py-4 whitespace-nowrap">{row.commodity}</td>
              <td className="px-6 py-4 whitespace-nowrap">{row.market}</td>
              <td className="px-6 py-4 whitespace-nowrap">{row.region}</td>
              <td className="px-6 py-4 whitespace-nowrap">₹{row.currentPrice}</td>
              <td className="px-6 py-4 whitespace-nowrap">₹{row.predictedPrice}</td>
              <td className="px-6 py-4 whitespace-nowrap flex items-center">
                <span className={`flex items-center ${row.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {row.change >= 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
                  {Math.abs(row.change)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DataTable;