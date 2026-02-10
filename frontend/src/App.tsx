import { RouterProvider } from 'react-router';
import { router } from './routes';
import { Toaster } from './components/ui/sonner';
import { LanguageProvider } from './contexts/LanguageContext';

export default function App() {
  return (
    <LanguageProvider>
      <RouterProvider
        router={router}
        fallbackElement={
          <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
            页面加载中...
          </div>
        }
      />
      <Toaster />
    </LanguageProvider>
  );
}
