import React from 'react';
import { Search, Calendar } from 'lucide-react';

const FilterSelect = ({ label, options }: { label: string; options: string[] }) => (
  <div className="w-full">
    <label className="block text-lg font-medium text-gray-800 mb-2">{label}</label>
    <select className="w-full p-3 bg-white border-2 border-gray-200 rounded-lg focus:border-green-500 focus:ring-0 transition-colors text-gray-700">
      <option value="">Select {label}</option>
      {options.map((option) => (
        <option key={option} value={option}>{option}</option>
      ))}
    </select>
  </div>
);

const CommodityFilter = () => {
  return (
    <div className="bg-gray-50 p-8 rounded-xl shadow-lg">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        <FilterSelect 
          label="Commodity" 
          options={['Wheat', 'Rice', 'Corn', 'Soybean','atta','Gram', 'tur', 'urad', 'moong', 'masur','Groundnut oil', 'mustard oil', 'vanaspati', 'soya oil', 'sunflower oil', 'palm oil', 'Onion', 'Tomato','Sugar', 'gur', 'milk', 'tea', 'salt' ]} 
        />
        <FilterSelect 
          label="Market Location" 
          options={['Kurnool', 'Madanapalle', 'Hyderabad', 'Mangalagiri']} 
        />
        <div className="w-full">
          <label className="block text-lg font-medium text-gray-800 mb-2">Date Range</label>
          <div className="flex space-x-4">
            <input 
              type="date" 
              className="flex-1 p-3 bg-white border-2 border-gray-200 rounded-lg focus:border-green-500 focus:ring-0 transition-colors"
            />
            <input 
              type="date" 
              className="flex-1 p-3 bg-white border-2 border-gray-200 rounded-lg focus:border-green-500 focus:ring-0 transition-colors"
            />
          </div>
        </div>
      </div>
      <div className="mt-8 flex justify-center">
        <button className="flex items-center space-x-2 bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors text-lg">
          <Search className="h-5 w-5" />
          <span>Search Markets</span>
        </button>
      </div>
    </div>
  );
};

export default CommodityFilter;