"use client";

import { FC } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import {
  ChatBubbleLeftRightIcon,
  FolderIcon,
  LinkIcon,
  PlusIcon,
  DocumentDuplicateIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Chat', href: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Projects', href: '/projects', icon: FolderIcon },
  { name: 'Connections', href: '/connections', icon: LinkIcon },
];

const shortcuts = [
  { name: 'Add Quote', href: '/quotes/add', icon: PlusIcon },
  { name: 'Compare Quotes', href: '/quotes/compare', icon: DocumentDuplicateIcon },
];

const Sidebar: FC = () => {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col bg-white px-3 py-4 shadow-sm">
      <div className="mb-8">
        <Link href="/" className="flex items-center gap-2 px-2">
          <span className="text-xl font-semibold">SteraFlow</span>
        </Link>
      </div>

      <div className="space-y-8">
        <nav className="flex flex-col">
          <div className="text-sm font-medium text-gray-500 px-2 mb-2">Navigation</div>
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={clsx(
                  'flex items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium',
                  isActive
                    ? 'bg-gray-100 text-gray-900'
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <nav className="flex flex-col">
          <div className="text-sm font-medium text-gray-500 px-2 mb-2">Shortcuts</div>
          {shortcuts.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className="flex items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900"
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          ))}
        </nav>
      </div>

      <div className="mt-auto">
        <div className="flex items-center gap-2 px-2 py-2">
          <div className="h-8 w-8 rounded-full bg-gray-200" />
          <div className="text-sm font-medium">John Doe</div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 