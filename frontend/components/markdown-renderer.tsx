// "use client";
//
// import React from "react";
// import ReactMarkdown from "react-markdown";
// import remarkGfm from "remark-gfm";
//
// type Props = {
//   content: string;
//   className?: string;
// };
//
// export function MarkdownRenderer({ content, className = "" }: Props) {
//   // use `any` to avoid strict typing mismatch in react-markdown v10
//   const components: any = {
//     p: ({ children }: any) => <p className="mb-2 last:mb-0">{children}</p>,
//     a: ({ children, ...props }: any) => (
//       <a {...props} className="text-primary underline">
//         {children}
//       </a>
//     ),
//     ul: ({ children }: any) => <ul className="list-disc ml-6">{children}</ul>,
//     ol: ({ children }: any) => <ol className="list-decimal ml-6">{children}</ol>,
//     blockquote: ({ children }: any) => (
//       <blockquote className="border-l-4 border-muted pl-4 italic">
//         {children}
//       </blockquote>
//     ),
//     code: ({ inline, children }: any) =>
//       inline ? (
//         <code className="bg-muted rounded px-1 py-0.5 text-sm font-mono">
//           {children}
//         </code>
//       ) : (
//         <pre className="bg-muted p-3 rounded-lg overflow-x-auto">
//           <code>{children}</code>
//         </pre>
//       ),
//   };
//
//   return (
//     <div className={`prose prose-sm max-w-none dark:prose-invert ${className}`}>
//       <ReactMarkdown remarkPlugins={[remarkGfm]} components={components as any}>
//         {content}
//       </ReactMarkdown>
//     </div>
//   );
// }


"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check } from "lucide-react";
import { useState } from "react";

type Props = {
  content: string;
  className?: string;
};

export function MarkdownRenderer({ content, className = "" }: Props) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCode(text);
      setTimeout(() => setCopiedCode(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  // use `any` to avoid strict typing mismatch in react-markdown v10
  const components: any = {
    p: ({ children }: any) => (
      <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
    ),

    a: ({ children, href, ...props }: any) => (
      <a
        {...props}
        href={href}
        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline underline-offset-2 hover:underline-offset-4 transition-all duration-200"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),

    ul: ({ children }: any) => (
      <ul className="list-disc ml-6 mb-3 space-y-1">{children}</ul>
    ),

    ol: ({ children }: any) => (
      <ol className="list-decimal ml-6 mb-3 space-y-1">{children}</ol>
    ),

    li: ({ children }: any) => (
      <li className="leading-relaxed">{children}</li>
    ),

    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-blue-500 dark:border-blue-400 bg-blue-50/50 dark:bg-blue-900/20 pl-4 py-2 my-3 rounded-r-lg italic">
        {children}
      </blockquote>
    ),

    h1: ({ children }: any) => (
      <h1 className="text-2xl font-bold mb-4 mt-6 first:mt-0 text-slate-900 dark:text-slate-100 border-b border-slate-200 dark:border-slate-700 pb-2">
        {children}
      </h1>
    ),

    h2: ({ children }: any) => (
      <h2 className="text-xl font-semibold mb-3 mt-5 first:mt-0 text-slate-800 dark:text-slate-200">
        {children}
      </h2>
    ),

    h3: ({ children }: any) => (
      <h3 className="text-lg font-medium mb-2 mt-4 first:mt-0 text-slate-700 dark:text-slate-300">
        {children}
      </h3>
    ),

    table: ({ children }: any) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700 border border-slate-200 dark:border-slate-700 rounded-lg">
          {children}
        </table>
      </div>
    ),

    thead: ({ children }: any) => (
      <thead className="bg-slate-50 dark:bg-slate-800">
        {children}
      </thead>
    ),

    th: ({ children }: any) => (
      <th className="px-4 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
        {children}
      </th>
    ),

    td: ({ children }: any) => (
      <td className="px-4 py-3 text-sm text-slate-600 dark:text-slate-400 border-t border-slate-200 dark:border-slate-700">
        {children}
      </td>
    ),

    code: ({ inline, children, className: codeClassName, ...props }: any) => {
      const match = /language-(\w+)/.exec(codeClassName || '');
      const language = match ? match[1] : '';
      const codeText = String(children).replace(/\n$/, '');

      if (inline) {
        return (
          <code className="bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-md px-2 py-1 text-sm font-mono border border-slate-200 dark:border-slate-700">
            {children}
          </code>
        );
      }

      return (
        <div className="relative group my-4">
          <div className="flex items-center justify-between bg-slate-800 dark:bg-slate-900 text-slate-300 px-4 py-2 text-sm rounded-t-lg border-b border-slate-700">
            <span className="font-medium">
              {language ? language.toUpperCase() : 'CODE'}
            </span>
            <button
              onClick={() => copyToClipboard(codeText)}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs hover:bg-slate-700 dark:hover:bg-slate-800 transition-colors duration-200"
              title="Copy to clipboard"
            >
              {copiedCode === codeText ? (
                <>
                  <Check className="w-3 h-3" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
          <pre className="bg-slate-50 dark:bg-slate-900 p-4 rounded-b-lg overflow-x-auto border border-slate-200 dark:border-slate-700 border-t-0">
            <code className="text-slate-800 dark:text-slate-200 text-sm font-mono leading-relaxed">
              {children}
            </code>
          </pre>
        </div>
      );
    },

    hr: () => (
      <hr className="my-6 border-t border-slate-200 dark:border-slate-700" />
    ),

    strong: ({ children }: any) => (
      <strong className="font-semibold text-slate-900 dark:text-slate-100">
        {children}
      </strong>
    ),

    em: ({ children }: any) => (
      <em className="italic text-slate-700 dark:text-slate-300">
        {children}
      </em>
    ),
  };

  return (
    <div className={`prose prose-sm max-w-none leading-relaxed ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components as any}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
