# ResepAI — Frontend Implementation Brief

## Project Location
`/home/ubuntu/recipe-chatbot/frontend/`

## What Already Exists
- `package.json` — React + Vite + react-markdown + remark-gfm + react-icons + shadcn deps (class-variance-authority, clsx, tailwind-merge, tailwindcss-animate, tailwindcss, postcss, autoprefixer)
- `vite.config.js` — Vite config with /api proxy to localhost:3001
- `index.html` — Entry HTML
- `src/main.jsx` — React entry point
- `src/App.jsx` — Complete chat UI (ThemeProvider, Header, MessageList, MessageBubble, ChatInput, TypingIndicator)

## What You Need to Create

### 1. `tailwind.config.js`
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
        },
      },
    },
  },
  plugins: [],
}
```

### 2. `postcss.config.js`
```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

### 3. `src/styles/global.css`
Use Tailwind directives (@tailwind base/components/utilities). Also include:
- CSS variables for theming under `:root` and `[data-theme="dark"]`
- Light theme defaults, dark theme overrides
- Smooth transitions for theme switching
- Custom styles for chat bubbles, typing indicator, welcome screen, example chips
- Use react-icons (FiSend, FiSun, FiMoon, FiChefHat or similar)

### 4. `src/lib/utils.js`
```js
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
```

### 5. Update `src/App.jsx`
- Import react-icons: `import { FiSend, FiSun, FiMoon } from 'react-icons/fi'`
- Replace emoji buttons with FiSun/FiMoon for theme toggle, FiSend for send button
- Keep all existing logic intact, only swap emojis for icons

### 6. `public/favicon.svg`
Simple cooking pot or chef hat SVG icon.

## Design Specs
- **Primary accent:** Blue — #3B82F6 (light) / #60A5FA (dark)
- **Layout:** Fixed header, scrollable message area, fixed bottom input
- **Chat bubbles:** User (right, blue bg), Bot (left, gray bg)
- **Welcome screen:** Title, description, 3 example chips
- **Typing indicator:** 3 bouncing dots animation
- **Responsive:** Mobile-friendly
- **Theme toggle:** Sun/moon icons in header

## After Creating Files
1. Run: `cd /home/ubuntu/recipe-chatbot/frontend && npm install`
2. Run: `npm run build`
3. If build succeeds: `cd /home/ubuntu/recipe-chatbot && git init && git add -A && git commit -m 'feat: ResepAI frontend — React + Vite + Tailwind + shadcn + react-icons'`

## Important
- Do NOT change App.jsx logic — only swap emojis for react-icons
- Keep styling clean and minimal
- Use Tailwind CSS with shadcn-style utility classes (cn() helper)
