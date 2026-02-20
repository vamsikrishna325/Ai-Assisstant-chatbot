import { useState, useEffect, useRef } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { FormattedMessage } from './FormattedMessage';
import { Send, Upload, Camera, Plus, Sparkles, User, Trash2, Share2, MoreVertical, Bell, Megaphone, X, Menu, CheckCircle2, Clock, AlertCircle } from 'lucide-react';
import { auth, googleProvider, db } from './firebase';
import { supabase } from './supabase';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  type User as FirebaseUser
} from 'firebase/auth';
import {
  collection,
  addDoc,
  updateDoc,
  deleteDoc,
  doc,
  query,
  where,
  onSnapshot,
  serverTimestamp,
  Timestamp
} from 'firebase/firestore';
import './App.css';
const FLASK_API_URL = 'http://localhost:5000/api';

interface Message {
  text: string;
  sender: string;
  timestamp?: number;
  fileUrl?: string;
  fileType?: string;
  fileName?: string;
}
interface Reminder {
  id: string;
  title: string;
  time: string;
  done: boolean;
}

interface Announcement {
  id: string;
  title: string;
  body: string;
  type: 'info' | 'warning' | 'success';
  read: boolean;
  event_data?: {
    event_name?: string;
    event_type?: string;
    department?: string;
    venue?: string;
    time?: string;
    description?: string;
    dates?: {
      event_date?: string;
      registration_deadline?: string;
      end_date?: string;
    };
    topics?: string[];
    registration_fee?: string;
    prizes?: string;
    contact_persons?: any[];
    additional_info?: string;
  };
  processed_at?: string;
}
interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

interface ChatHistoryItem {
  id: string;
  title: string;
  date: string;
  messages: Message[];
  userId: string;
  createdAt: any;
  updatedAt: any;
}

type SidebarView = 'chat' | 'reminders' | 'announcements';

export default function ChatbotInterface() {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [isSignUp, setIsSignUp] = useState<boolean>(false);
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [fullName, setFullName] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [sidebarView, setSidebarView] = useState<SidebarView>('chat');
  const [showChatSidebar, setShowChatSidebar] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const shouldScrollRef = useRef<boolean>(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);  
  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [reminders, setReminders] = useState<Reminder[]>([
    { id: '1', title: 'Review study notes', time: '', done: false },
    { id: '2', title: 'Submit assignment by 5 PM', time: '', done: false },
  ]);

const notifiedAnnouncementIds = useRef<Set<string>>(new Set());
  const [selectedAnnouncement, setSelectedAnnouncement] = useState<Announcement | null>(null);
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [reminderInput, setReminderInput] = useState('');
  const [reminderTime, setReminderTime] = useState('');
 const [pendingFiles, setPendingFiles] = useState<File[]>([]);
 const [pendingFilePreviews, setPendingFilePreviews] = useState<string[]>([]);
  const scrollToBottom = () => {
    if (shouldScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };
  // ── Toast helper ─────────────────────────────────────────────────────────
  const showToast = (message: string, type: Toast['type'] = 'info') => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500);
  };

  // ── Reminder actions ──────────────────────────────────────────────────────
  const addReminder = () => {
    if (!reminderInput.trim()) return;
    setReminders(prev => [{ id: Date.now().toString(), title: reminderInput.trim(), time: reminderTime, done: false }, ...prev]);
    setReminderInput('');
    setReminderTime('');
    showToast('✅ Reminder added!', 'success');
  };
  const toggleReminder = (id: string) => setReminders(prev => prev.map(r => r.id === id ? { ...r, done: !r.done } : r));
  const deleteReminder  = (id: string) => { setReminders(prev => prev.filter(r => r.id !== id)); showToast('Reminder deleted', 'info'); };

const viewAnnouncementDetails = (announcement: Announcement) => {
  markRead(announcement.id);
  setSelectedAnnouncement(announcement);
};
const closeAnnouncementDetail = (): void => {
  setSelectedAnnouncement(null)
};
  // ── Announcement actions ───────────────────────────────────────────────────
  const markRead    = (id: string) => setAnnouncements(prev => prev.map(a => a.id === id ? { ...a, read: true } : a));
  const markAllRead = () => { setAnnouncements(prev => prev.map(a => ({ ...a, read: true }))); showToast('All marked as read', 'success'); };

  // ── Badge counts ───────────────────────────────────────────────────────────
  const unreadCount   = announcements.filter(a => !a.read).length;
  const pendingCount  = reminders.filter(r => !r.done).length;

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  // Request notification permission on mount
useEffect(() => {
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }
}, []);

// Check reminders every minute for notifications
// Check reminders every minute for notifications
// Check reminders every minute for notifications
useEffect(() => {
  const notifiedReminderIds = new Set<string>();

  const checkReminders = () => {
    const now = new Date();

    reminders.forEach(reminder => {
      if (!reminder.done && reminder.time && !notifiedReminderIds.has(reminder.id)) {
        const reminderDate = new Date(reminder.time);

        // Match exact date + hour + minute
        const sameYear   = now.getFullYear()  === reminderDate.getFullYear();
        const sameMonth  = now.getMonth()     === reminderDate.getMonth();
        const sameDay    = now.getDate()      === reminderDate.getDate();
        const sameHour   = now.getHours()     === reminderDate.getHours();
        const sameMinute = now.getMinutes()   === reminderDate.getMinutes();

        if (sameYear && sameMonth && sameDay && sameHour && sameMinute) {
          notifiedReminderIds.add(reminder.id);

          const timeLabel = reminderDate.toLocaleString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          });

          if (Notification.permission === 'granted') {
            new Notification('⏰ Reminder', {
              body: `${reminder.title}\n🕐 ${timeLabel}`,
              icon: '/favicon.ico'
            });
          }

          showToast(`⏰ ${reminder.title} — ${timeLabel}`, 'info');
        }
      }
    });
  };

  checkReminders(); // run immediately on mount
  const interval = setInterval(checkReminders, 60000);
  return () => clearInterval(interval);
}, [reminders]);

useEffect(() => {
  if (isCameraOpen && videoRef.current && streamRef.current) {
    videoRef.current.srcObject = streamRef.current;
  }
}, [isCameraOpen]);
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (!currentUser) {
        setChatHistory([]);
        setMessages([]);
        setCurrentChatId(null);
      }
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (!user) return;

    const q = query(
      collection(db, 'chats'),
      where('userId', '==', user.uid)
    );

    const unsubscribe = onSnapshot(q, 
      (snapshot) => {
        const chats: ChatHistoryItem[] = [];
        snapshot.forEach((docSnap) => {
          const data = docSnap.data();
          
          chats.push({
            id: docSnap.id,
            title: data.title || 'Untitled Chat',
            date: formatDate(data.updatedAt),
            messages: data.messages || [],
            userId: data.userId,
            createdAt: data.createdAt,
            updatedAt: data.updatedAt
          });
        });
        
        chats.sort((a, b) => {
          const aTime = a.updatedAt?.toMillis?.() || 0;
          const bTime = b.updatedAt?.toMillis?.() || 0;
          return bTime - aTime;
        });
        
        setChatHistory(chats);
      },
      (error) => {
        console.error('Error loading chat history:', error);
      }
    );

    return () => unsubscribe();
  }, [user]);
 useEffect(() => {
    const notifiedIds = new Set<string>(); // tracks already-notified announcements

    const fetchAnnouncements = async () => {
      try {
        const response = await fetch(`${FLASK_API_URL}/announcements/latest?limit=10`);
        const data = await response.json();

        if (data.success && data.announcements) {
          const transformedAnnouncements = data.announcements.map((ann: any) => ({
            id: ann.id,
            title: ann.title,
            body: ann.body,
            type: ann.type,
            read: false,
            event_data: ann.event_data,
            processed_at: ann.processed_at
          }));

          setAnnouncements(transformedAnnouncements);

          // Only notify for NEW announcements not seen before
          transformedAnnouncements.forEach((ann: Announcement) => {
            if (!notifiedIds.has(ann.id)) {
              notifiedIds.add(ann.id);

              // Skip notification on first load (populate silently)
              // Only notify after initial fetch is done
              if (notifiedIds.size > transformedAnnouncements.length) {
                if (Notification.permission === 'granted') {
                  new Notification('📢 New Announcement', {
                    body: ann.title,
                    icon: '/favicon.ico'
                  });
                }
                showToast(`📢 ${ann.title}`, ann.type === 'warning' ? 'error' : ann.type === 'success' ? 'success' : 'info');
              }
            }
          });
        }
      } catch (error) {
        console.error('Error fetching announcements:', error);
      }
    };

    fetchAnnouncements();
    const interval = setInterval(fetchAnnouncements, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (timestamp: any) => {
    if (!timestamp) return 'Just now';
    
    try {
      const date = timestamp instanceof Timestamp ? timestamp.toDate() : new Date(timestamp);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const minutes = Math.floor(diff / (1000 * 60));
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));

      if (minutes < 1) return 'Just now';
      if (minutes < 60) return `${minutes}m ago`;
      if (hours < 24) return `${hours}h ago`;
      if (days === 0) return 'Today';
      if (days === 1) return 'Yesterday';
      if (days < 7) return `${days}d ago`;
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch (e) {
      return 'Recently';
    }
  };

  const handleLogin = async (): Promise<void> => {
    if (!email || !password) {
      setError('Please enter email and password');
      return;
    }
    
    setError('');
    setLoading(true);
    
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      setEmail('');
      setPassword('');
      setFullName('');
    } catch (err: any) {
      if (err.code === 'auth/email-already-in-use') {
        setError('Email already in use. Please sign in instead.');
      } else if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        setError('Incorrect password. Please try again.');
      } else if (err.code === 'auth/user-not-found') {
        setError('No account found. Please sign up first.');
      } else if (err.code === 'auth/weak-password') {
        setError('Password should be at least 6 characters.');
      } else if (err.code === 'auth/invalid-email') {
        setError('Invalid email address.');
      } else {
        setError(err.message || 'Authentication failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async (): Promise<void> => {
    setError('');
    setLoading(true);
    
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err: any) {
      setError(err.message || 'Google sign-in failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async (): Promise<void> => {
    try {
      await signOut(auth);
      setMessages([]);
      setCurrentChatId(null);
    } catch (err: any) {
      console.error('Logout error:', err);
    }
  };

  // Fixed: Upload file to Supabase Storage
  const uploadFileToStorage = async (file: File): Promise<string> => {
    if (!user) throw new Error('No user logged in');
    
    const fileName = `${user.uid}/${Date.now()}_${file.name}`;
    
    const { data, error } = await supabase.storage
      .from('test')
      .upload(fileName, file, {
        cacheControl: '3600',
        upsert: false
      });

    if (error) {
      console.error('Supabase upload error:', error);
      throw new Error(`Upload failed: ${error.message}`);
    }

    const { data: urlData } = supabase.storage
      .from('test')
      .getPublicUrl(fileName);

    return urlData.publicUrl;
  };

  const saveChatToFirestore = async (chatMessages: Message[], chatId?: string | null) => {
    if (!user || chatMessages.length === 0) return null;

    const firstMessage = chatMessages[0].text.slice(0, 50);
    const title = firstMessage.length < 50 ? firstMessage : firstMessage + '...';

    try {
      if (chatId) {
        const chatRef = doc(db, 'chats', chatId);
        await updateDoc(chatRef, {
          messages: chatMessages,
          title: title,
          updatedAt: serverTimestamp()
        });
        return chatId;
      } else {
        const docRef = await addDoc(collection(db, 'chats'), {
          title: title,
          messages: chatMessages,
          userId: user.uid,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        });
        return docRef.id;
      }
    } catch (err) {
      console.error('Error saving chat:', err);
      return null;
    }
  };

  const handleSend = async (): Promise<void> => {
  if (!user) {
    showToast('Please sign in to chat', 'error');
    return;
  }

 
  // Handle file upload with query
  if (pendingFiles.length > 0) {
    setUploading(true);
    shouldScrollRef.current = true;

    try {
      const queryText = input.trim() || '';
      let updatedMessages = [...messages];

      for (const file of pendingFiles) {
        const fileType = file.type.startsWith('image/') ? 'image' : 'file';

        // Upload to Supabase for display in chat
        const fileUrl = await uploadFileToStorage(file);

        const userMessage: Message = {
          text: queryText || `Uploaded: ${file.name}`,
          sender: 'user',
          timestamp: Date.now(),
          fileUrl,
          fileType,
          fileName: file.name
        };
        updatedMessages = [...updatedMessages, userMessage];
        setMessages([...updatedMessages]);

        // Send file to Flask for RAG processing
        const formData = new FormData();
        formData.append('file', file);
        if (queryText) formData.append('message', queryText);

        setIsTyping(true);
        const flaskResponse = await fetch(`${FLASK_API_URL}/upload`, {
          method: 'POST',
          body: formData
        });

        if (!flaskResponse.ok) throw new Error(`Flask upload failed: ${flaskResponse.status}`);

        const flaskData = await flaskResponse.json();
        setIsTyping(false);

        const botMessage: Message = {
          text: flaskData.response || 'File processed successfully.',
          sender: 'bot',
          timestamp: Date.now()
        };

        updatedMessages = [...updatedMessages, botMessage];
        setMessages([...updatedMessages]);
      }

      setInput('');
      setPendingFiles([]);
      setPendingFilePreviews([]);

      const savedChatId = await saveChatToFirestore(updatedMessages, currentChatId);
      if (!currentChatId && savedChatId) setCurrentChatId(savedChatId);

      if (currentChatId) {
        await updateDoc(doc(db, 'chats', currentChatId), {
          messages: updatedMessages,
          updatedAt: serverTimestamp()
        });
      }

    } catch (error: any) {
      console.error('Error uploading file:', error);
      showToast(`Failed to upload file: ${error.message}`, 'error');
      setIsTyping(false);
    } finally {
      setUploading(false);
    }
    return;
  }

  // Regular text message handling
  if (!input.trim()) return;
  
  const userMessage: Message = {
    text: input.trim(),
    sender: 'user',
    timestamp: Date.now()
  };
  
  const updatedMessages = [...messages, userMessage];
  setMessages(updatedMessages);
  setInput('');
  setIsTyping(true);
  
  try {
    const response = await fetch(`${FLASK_API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: userMessage.text
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    const botMessage: Message = {
      text: data.response || 'Sorry, I could not process your request.',
      sender: 'bot',
      timestamp: Date.now()
    };
    
    const messagesWithBot = [...updatedMessages, botMessage];
    setMessages(messagesWithBot);
    setIsTyping(false);
    
    if (currentChatId) {
      const chatRef = doc(db, 'chats', currentChatId);
      await updateDoc(chatRef, {
        messages: messagesWithBot,
        updatedAt: serverTimestamp()
      });
    } else {
      const title = userMessage.text.slice(0, 50) + (userMessage.text.length > 50 ? '...' : '');
      const newChatRef = await addDoc(collection(db, 'chats'), {
        title: title,
        messages: messagesWithBot,
        userId: user.uid,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      });
      setCurrentChatId(newChatRef.id);
    }
    
  } catch (error) {
    console.error('Error sending message:', error);
    setIsTyping(false);
    
    const errorMessage: Message = {
      text: 'Sorry, I encountered an error connecting to the chatbot service. Please make sure the Flask server is running.',
      sender: 'bot',
      timestamp: Date.now()
    };
    
    const messagesWithError = [...updatedMessages, errorMessage];
    setMessages(messagesWithError);
    
    if (currentChatId) {
      const chatRef = doc(db, 'chats', currentChatId);
      await updateDoc(chatRef, {
        messages: messagesWithError,
        updatedAt: serverTimestamp()
      });
    }
  }
};

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
  const files = Array.from(e.target.files || []);
  if (files.length === 0) return;

  const validFiles: File[] = [];
  const previews: string[] = [];

  for (const file of files) {
    if (file.size > 10 * 1024 * 1024) {
      showToast(`❌ ${file.name} exceeds 10MB limit`, 'error');
      continue;
    }
    validFiles.push(file);
  }

  if (validFiles.length === 0) {
    e.target.value = '';
    return;
  }

  // Generate previews
  await Promise.all(validFiles.map((file, i) =>
    new Promise<void>((resolve) => {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          previews[i] = ev.target?.result as string;
          resolve();
        };
        reader.readAsDataURL(file);
      } else {
        previews[i] = '';
        resolve();
      }
    })
  ));

  setPendingFiles(prev => [...prev, ...validFiles]);
  setPendingFilePreviews(prev => [...prev, ...previews]);
  showToast(`📎 ${validFiles.length} file(s) attached`, 'success');
  e.target.value = '';
};

const removePendingFile = (index: number): void => {
  setPendingFiles(prev => prev.filter((_, i) => i !== index));
  setPendingFilePreviews(prev => prev.filter((_, i) => i !== index));
};


const openCamera = async (): Promise<void> => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streamRef.current = stream;   
    setIsCameraOpen(true);       
  } catch (err) {
    console.error('Camera access error:', err);
    alert('Could not access camera');
  }
};


const closeCamera = (): void => {
  if (streamRef.current) {
    streamRef.current.getTracks().forEach(track => track.stop());
    streamRef.current = null;       
  }
  if (videoRef.current) {
    videoRef.current.srcObject = null;
  }
  setIsCameraOpen(false);
};

  const capturePhoto = async (): Promise<void> => {
  if (!videoRef.current) return;

  const canvas = document.createElement('canvas');
  canvas.width = videoRef.current.videoWidth;
  canvas.height = videoRef.current.videoHeight;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    ctx.drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `photo_${Date.now()}.jpg`, { type: 'image/jpeg' });
        
        setPendingFiles(prev => [...prev, file]);
const previewUrl = canvas.toDataURL('image/jpeg');
setPendingFilePreviews(prev => [...prev, previewUrl])
        
        showToast('📷 Photo captured! Add a message and send.', 'success');
        closeCamera();  // close camera, return to chat input
      }
    }, 'image/jpeg');
  }
};

  const handleNewChat = (): void => {
    setMessages([]);
    setCurrentChatId(null);
    shouldScrollRef.current = true;
    if (window.innerWidth <= 768) {
      setShowChatSidebar(false);
    }
  };

  const handleSelectChat = (chat: ChatHistoryItem) => {
    shouldScrollRef.current = false;
    setMessages(chat.messages);
    setCurrentChatId(chat.id);
    setOpenMenuId(null);
    if (window.innerWidth <= 768) {
      setShowChatSidebar(false);
    }
    setTimeout(() => {
      shouldScrollRef.current = true;
    }, 100);
  };

  const handleDeleteChat = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenMenuId(null);
    
    if (!window.confirm('Are you sure you want to delete this chat?')) return;

    try {
      await deleteDoc(doc(db, 'chats', chatId));
      
      if (currentChatId === chatId) {
        setMessages([]);
        setCurrentChatId(null);
      }
    } catch (err) {
      console.error('Error deleting chat:', err);
      alert('Failed to delete chat');
    }
  };

  const handleShareChat = async (chat: ChatHistoryItem, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenMenuId(null);
    
    const chatText = chat.messages
      .map(msg => `${msg.sender === 'user' ? 'You' : 'Bot'}: ${msg.text}`)
      .join('\n\n');
    
    try {
      if (navigator.share) {
        await navigator.share({
          title: chat.title,
          text: chatText
        });
      } else {
        await navigator.clipboard.writeText(chatText);
        alert('Chat copied to clipboard!');
      }
    } catch (err) {
      console.error('Error sharing chat:', err);
    }
  };

  const toggleMenu = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenMenuId(openMenuId === chatId ? null : chatId);
  };

  const suggestedPrompts = [
    'Explain quantum computing',
    'Help me write an email',
    'Analyze this image',
    'Generate ideas'
  ];

  if (!user) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h1 className="auth-title">
            {isSignUp ? 'Create Account' : 'Welcome Back'}
          </h1>
          <p className="auth-subtitle">
            {isSignUp ? 'Sign up to get started' : 'Sign in to continue'}
          </p>
          
          {error && <div className="auth-error">{error}</div>}
          
          <div className="auth-form">
            {isSignUp && (
              <input
                type="text"
                placeholder="Full Name"
                className="auth-input"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            )}
            <input
              type="email"
              placeholder="Email"
              className="auth-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="Password (min 6 characters)"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button 
              onClick={handleLogin} 
              className="auth-button"
              disabled={loading}
            >
              {loading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}
            </button>

            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button 
              onClick={handleGoogleSignIn} 
              className="google-button"
              disabled={loading}
            >
              <svg width="18" height="18" viewBox="0 0 18 18">
                <path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18z"/>
                <path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17z"/>
                <path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 0 1 0-3.04V5.41H1.83a8 8 0 0 0 0 7.18l2.67-2.07z"/>
                <path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.4L4.5 7.49a4.77 4.77 0 0 1 4.48-3.3z"/>
              </svg>
              Continue with Google
            </button>
          </div>
          
          <p className="auth-toggle">
            {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
            <button onClick={() => setIsSignUp(!isSignUp)} className="auth-link">
              {isSignUp ? 'Sign In' : 'Sign Up'}
            </button>
          </p>
        </div>
      </div>
    );
  }

  const isMobile = window.innerWidth <= 768;

  return (
    <div className="chat-layout">
      {/* Mobile overlay */}
      {showChatSidebar && isMobile && (
        <div 
          className="sidebar-overlay active"
          onClick={() => setShowChatSidebar(false)}
        />
      )}
      
      {/* Chat History Sidebar */}
      <div className={`sidebar-dark ${showChatSidebar && isMobile ? 'mobile-open' : ''}`} style={{ display: !showChatSidebar && !isMobile ? 'none' : 'flex' }}>
        <button className="new-chat-btn" onClick={handleNewChat}>
          <Plus size={18} />
          New Chat
        </button>
        
        <div className="chat-history-list">
          {chatHistory.length === 0 ? (
            <div style={{ padding: '1rem', color: '#8b92a8', fontSize: '0.875rem', textAlign: 'center' }}>
              No chat history yet. Start a conversation!
            </div>
          ) : (
            chatHistory.map((chat) => (
              <div 
                key={chat.id} 
                className={`chat-history-item-dark ${currentChatId === chat.id ? 'active' : ''}`}
                onClick={() => handleSelectChat(chat)}
              >
                <div className="chat-item-content">
                  <span className="chat-item-title">{chat.title}</span>
                  <span className="chat-item-date">{chat.date}</span>
                </div>
                <div className="chat-item-menu">
                  <button 
                    onClick={(e) => toggleMenu(chat.id, e)}
                    className="chat-menu-btn"
                    title="More options"
                  >
                    <MoreVertical size={16} />
                  </button>
                  {openMenuId === chat.id && (
                    <div className="chat-menu-dropdown">
                      <button 
                        onClick={(e) => handleShareChat(chat, e)}
                        className="chat-menu-item"
                      >
                        <Share2 size={14} />
                        Share
                      </button>
                      <button 
                        onClick={(e) => handleDeleteChat(chat.id, e)}
                        className="chat-menu-item delete"
                      >
                        <Trash2 size={14} />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="user-profile">
          <div className="user-avatar">
            <User size={20} />
          </div>
          <div className="user-info">
            <span className="user-name">{user.email?.split('@')[0] || 'User'}</span>
            <button onClick={handleLogout} className="logout-link">Logout</button>
          </div>
        </div>
      </div>

      <div className="chat-area-dark">
        <div className="chat-header-dark">
          <div className="assistant-info">
            <button 
              className="hamburger-btn"
              onClick={() => setShowChatSidebar(!showChatSidebar)}
              title="Toggle chat history"
            >
              <Menu size={24} />
            </button>
            <div className="assistant-avatar">
              <Sparkles size={24} />
            </div>
            <div>
              <h2 className="assistant-name">AI Assistant</h2>
              <span className="assistant-status">
                <span className="status-dot"></span>
                Online
              </span>
            </div>
          </div>

          <div className="header-actions">
            <button
              className="header-action-btn"
              onClick={() => setSidebarView(sidebarView === 'reminders' ? 'chat' : 'reminders')}
              title="Reminders"
              style={{ position: 'relative' }}
            >
              <Bell size={20} />
              {pendingCount > 0 && (
                <span style={{ position:'absolute', top:2, right:2, background:'#ef4444', color:'#fff', borderRadius:'50%', fontSize:'0.6rem', width:14, height:14, display:'flex', alignItems:'center', justifyContent:'center', fontWeight:700 }}>
                  {pendingCount}
                </span>
              )}
            </button>
            <button
              className="header-action-btn"
              onClick={() => setSidebarView(sidebarView === 'announcements' ? 'chat' : 'announcements')}
              title="Announcements"
              style={{ position: 'relative' }}
            >
              <Megaphone size={20} />
              {unreadCount > 0 && (
                <span style={{ position:'absolute', top:2, right:2, background:'#ef4444', color:'#fff', borderRadius:'50%', fontSize:'0.6rem', width:14, height:14, display:'flex', alignItems:'center', justifyContent:'center', fontWeight:700 }}>
                  {unreadCount}
                </span>
              )}
            </button>
          </div>
        </div>

        <div className="content-with-sidebar">
          <div className="messages-area-dark">
            {messages.length === 0 ? (
              <div className="welcome-screen" style={{ left: showChatSidebar ? '0px' : '0px' }}>
                <div className="welcome-icon">
                  <Sparkles size={48} />
                </div>
                <h1 className="welcome-title">How can I help you today?</h1>
                <p className="welcome-subtitle">
                  I'm your AI assistant. Ask me anything, share files, or take a photo to get started.
                </p>
                
                <div className="suggested-prompts">
                  {suggestedPrompts.map((prompt, idx) => (
                    <button 
                      key={idx} 
                      className="prompt-button"
                      onClick={() => setInput(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="messages-list">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message-row ${msg.sender}`}>
                    <div className={`message-bubble ${msg.sender}`}>
                      {msg.fileUrl && msg.fileType === 'image' && (
                        <img src={msg.fileUrl} alt={msg.fileName} style={{ maxWidth: '100%', borderRadius: '0.5rem', marginBottom: '0.5rem' }} />
                      )}
                      {msg.fileUrl && msg.fileType === 'file' && (
                        <a href={msg.fileUrl} target="_blank" rel="noopener noreferrer" style={{ color: 'inherit', textDecoration: 'underline' }}>
                          📎 {msg.fileName}
                        </a>
                      )}
                      {msg.sender === 'bot' ? (
                         <FormattedMessage text={msg.text} />
                             ) : (
                           <div>{msg.text}</div>
                            )}
                    </div>
                  </div>
                ))}
                {isTyping && (                                    
                   <div className="message-row bot">
                     <div className="message-bubble bot">
                       <div className="typing-indicator">
                         <span></span>
                         <span></span>
                         <span></span>
                       </div>
                     </div>
                   </div>
                )}      
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Right Sidebar for Reminders / Announcements */}
          {(sidebarView === 'reminders' || sidebarView === 'announcements') && (
            <div className="right-sidebar">
              <div className="right-sidebar-header">
                <div style={{ display:'flex', alignItems:'center', gap:'0.6rem' }}>
                  {sidebarView === 'reminders' ? <Bell size={18}/> : <Megaphone size={18}/>}
                  <h3 style={{ margin:0, fontSize:'0.95rem' }}>
                    {sidebarView === 'reminders' ? 'Reminders' : 'Announcements'}
                  </h3>
                </div>
                <button onClick={() => setSidebarView('chat')} className="close-panel-btn"><X size={18}/></button>
              </div>

              {/* ── REMINDERS PANEL ── */}
              {sidebarView === 'reminders' && (
                <div className="right-sidebar-content" style={{ display:'flex', flexDirection:'column', gap:'0.75rem' }}>
                  {/* Add reminder form */}
                  <div style={{ display:'flex', flexDirection:'column', gap:'0.4rem', padding:'0.5rem', background:'rgba(255,255,255,0.04)', borderRadius:8 }}>
                    <input
                      value={reminderInput}
                      onChange={e => setReminderInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && addReminder()}
                      placeholder="Add a reminder..."
                      style={{ background:'rgba(255,255,255,0.07)', border:'1px solid rgba(255,255,255,0.1)', borderRadius:6, padding:'0.4rem 0.6rem', color:'#e2e8f0', fontSize:'0.85rem', outline:'none' }}
                    />
                    <input
                      type="date"
                      value={reminderTime ? reminderTime.split('T')[0] : ''}
                      onChange={e => {
                        const timePart = reminderTime ? reminderTime.split('T')[1] : '00:00';
                        setReminderTime(`${e.target.value}T${timePart}`);
                      }}
                      style={{ background:'rgba(255,255,255,0.07)', border:'1px solid rgba(255,255,255,0.1)', borderRadius:6, padding:'0.4rem 0.6rem', color:'#94a3b8', fontSize:'0.8rem', outline:'none', width:'100%' }}
                    />
                    <input
                      type="time"
                      value={reminderTime ? reminderTime.split('T')[1] : ''}
                      onChange={e => {
                        const datePart = reminderTime ? reminderTime.split('T')[0] : new Date().toISOString().split('T')[0];
                        setReminderTime(`${datePart}T${e.target.value}`);
                      }}
                      style={{ background:'rgba(255,255,255,0.07)', border:'1px solid rgba(255,255,255,0.1)', borderRadius:6, padding:'0.4rem 0.6rem', color:'#94a3b8', fontSize:'0.8rem', outline:'none', width:'100%' }}
                    />
                    <button
                      onClick={addReminder}
                      style={{ background:'#3b82f6', border:'none', borderRadius:6, padding:'0.4rem', color:'#fff', fontSize:'0.82rem', cursor:'pointer', fontWeight:600 }}
                    >
                      + Add Reminder
                    </button>
                  </div>

                  {/* Reminder list */}
                  {reminders.length === 0
                    ? <p style={{ color:'#8b92a8', textAlign:'center', fontSize:'0.85rem' }}>No reminders yet.</p>
                    : reminders.map(r => (
                        <div key={r.id} style={{ display:'flex', alignItems:'flex-start', gap:'0.5rem', padding:'0.5rem 0.6rem', background:'rgba(255,255,255,0.04)', borderRadius:8, opacity: r.done ? 0.5 : 1 }}>
                          <button onClick={() => toggleReminder(r.id)} style={{ background:'none', border:'none', cursor:'pointer', color: r.done ? '#22c55e' : '#6b7280', padding:0, marginTop:2, flexShrink:0 }}>
                            <CheckCircle2 size={16}/>
                          </button>
                          <div style={{ flex:1, minWidth:0 }}>
                            <p style={{ margin:0, fontSize:'0.85rem', color:'#e2e8f0', textDecoration: r.done ? 'line-through' : 'none', wordBreak:'break-word' }}>{r.title}</p>
                            {r.time && <p style={{ margin:'0.15rem 0 0', fontSize:'0.75rem', color:'#60a5fa', display:'flex', alignItems:'center', gap:3 }}><Clock size={11}/>{new Date(r.time).toLocaleString()}</p>}
                          </div>
                          <button onClick={() => deleteReminder(r.id)} style={{ background:'none', border:'none', cursor:'pointer', color:'#ef4444', padding:0, flexShrink:0 }}>
                            <X size={14}/>
                          </button>
                        </div>
                      ))
        
                  }
                </div>
              )}
              
              {/* ── ANNOUNCEMENTS PANEL ── */}
              {sidebarView === 'announcements' && (
                <div className="right-sidebar-content" style={{ display:'flex', flexDirection:'column', gap:'0.6rem' }}>
                  {unreadCount > 0 && (
                    <button onClick={markAllRead} style={{ background:'rgba(59,130,246,0.15)', border:'1px solid rgba(59,130,246,0.3)', borderRadius:6, padding:'0.35rem 0.6rem', color:'#60a5fa', fontSize:'0.8rem', cursor:'pointer', textAlign:'center' }}>
                      Mark all as read
                    </button>
                  )}
                  {announcements.length === 0
                    ? <p style={{ color:'#8b92a8', textAlign:'center', fontSize:'0.85rem' }}>No announcements.</p>
                    : announcements.map(a => (
                        <div key={a.id} onClick={() => viewAnnouncementDetails(a)} style={{ padding:'0.6rem 0.75rem', borderRadius:8, cursor:'pointer', background: a.read ? 'rgba(255,255,255,0.03)' : 'rgba(59,130,246,0.08)', border:`1px solid ${a.read ? 'rgba(255,255,255,0.06)' : 'rgba(59,130,246,0.25)'}` }}>
                          <div style={{ display:'flex', alignItems:'center', gap:'0.4rem', marginBottom:'0.25rem' }}>
                            {a.type === 'success' && <AlertCircle size={13} color="#22c55e"/>}
                            {a.type === 'info'    && <AlertCircle size={13} color="#60a5fa"/>}
                            {a.type === 'warning' && <AlertCircle size={13} color="#f59e0b"/>}
                            <span style={{ fontSize:'0.85rem', fontWeight: a.read ? 400 : 600, color:'#e2e8f0' }}>{a.title}</span>
                            {!a.read && <span style={{ marginLeft:'auto', width:7, height:7, borderRadius:'50%', background:'#3b82f6', flexShrink:0 }}/>}
                          </div>
                          <p style={{ margin:0, fontSize:'0.8rem', color:'#94a3b8', lineHeight:1.5 }}>{a.body}</p>
                        </div>
                      ))
                  }
                  
                </div>
              )}
            </div>
          )}
        </div>
            {/*─ ANNOUNCEMENT DETAIL MODAL ── */}
              {selectedAnnouncement && (
                <div style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'rgba(0, 0, 0, 0.85)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 10000,
                  padding: '1rem'
                }}>
                  <div style={{
                    background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
                    borderRadius: 12,
                    maxWidth: '600px',
                    width: '100%',
                    maxHeight: '80vh',
                    overflow: 'auto',
                    padding: '1.5rem',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.6)'
                  }}>
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '1rem' }}>
                      <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#e2e8f0', fontWeight: 600 }}>
                        {selectedAnnouncement.title}
                      </h2>
                      <button onClick={closeAnnouncementDetail} style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: 6,
                        padding: '0.4rem',
                        cursor: 'pointer',
                        color: '#ef4444'
                      }}>
                        <X size={18} />
                      </button>
                    </div>

                    {/* Event Details */}
                   {selectedAnnouncement.event_data && (
  selectedAnnouncement.event_data.event_type ||
  selectedAnnouncement.event_data.department ||
  selectedAnnouncement.event_data.venue ||
  selectedAnnouncement.event_data.description ||
  selectedAnnouncement.event_data.dates
) ? (
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {/* Event Type & Department */}
                        {(selectedAnnouncement.event_data.event_type || selectedAnnouncement.event_data.department) && (
                          <div style={{ padding: '0.75rem', background: 'rgba(59, 130, 246, 0.08)', borderRadius: 8, border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                            {selectedAnnouncement.event_data.event_type && (
                              <p style={{ margin: '0 0 0.3rem', fontSize: '0.9rem', color: '#60a5fa' }}>
                                <strong>🎯 Type:</strong> {selectedAnnouncement.event_data.event_type}
                              </p>
                            )}
                            {selectedAnnouncement.event_data.department && (
                              <p style={{ margin: 0, fontSize: '0.9rem', color: '#60a5fa' }}>
                                <strong>🏢 Department:</strong> {selectedAnnouncement.event_data.department}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Dates Section */}
                        {selectedAnnouncement.event_data.dates && Object.keys(selectedAnnouncement.event_data.dates).length > 0 && (
                          <div style={{ padding: '0.75rem', background: 'rgba(34, 197, 94, 0.08)', borderRadius: 8, border: '1px solid rgba(34, 197, 94, 0.2)' }}>
                            <p style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#22c55e', fontWeight: 600 }}>📅 Important Dates</p>
                            {selectedAnnouncement.event_data.dates.event_date && (
                              <p style={{ margin: '0.2rem 0', fontSize: '0.85rem', color: '#94a3b8' }}>
                                <strong>Event Date:</strong> {selectedAnnouncement.event_data.dates.event_date}
                              </p>
                            )}
                            {selectedAnnouncement.event_data.dates.registration_deadline && (
                              <p style={{ margin: '0.2rem 0', fontSize: '0.85rem', color: '#94a3b8' }}>
                                <strong>⏰ Registration Deadline:</strong> {selectedAnnouncement.event_data.dates.registration_deadline}
                              </p>
                            )}
                            {selectedAnnouncement.event_data.dates.end_date && (
                              <p style={{ margin: '0.2rem 0', fontSize: '0.85rem', color: '#94a3b8' }}>
                                <strong>End Date:</strong> {selectedAnnouncement.event_data.dates.end_date}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Venue & Time */}
                        {(selectedAnnouncement.event_data.venue || selectedAnnouncement.event_data.time) && (
                          <div style={{ padding: '0.75rem', background: 'rgba(168, 85, 247, 0.08)', borderRadius: 8, border: '1px solid rgba(168, 85, 247, 0.2)' }}>
                            {selectedAnnouncement.event_data.venue && (
                              <p style={{ margin: '0 0 0.3rem', fontSize: '0.85rem', color: '#c084fc' }}>
                                <strong>📍 Venue:</strong> {selectedAnnouncement.event_data.venue}
                              </p>
                            )}
                            {selectedAnnouncement.event_data.time && (
                              <p style={{ margin: 0, fontSize: '0.85rem', color: '#c084fc' }}>
                                <strong>🕒 Time:</strong> {selectedAnnouncement.event_data.time}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Description */}
                        {selectedAnnouncement.event_data.description && (
                          <div style={{ padding: '0.75rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 8, border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                            <p style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#e2e8f0', fontWeight: 600 }}>📋 Description</p>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                              {selectedAnnouncement.event_data.description}
                            </p>
                          </div>
                        )}

                        {/* Topics */}
                        {selectedAnnouncement.event_data.topics && selectedAnnouncement.event_data.topics.length > 0 && (
                          <div style={{ padding: '0.75rem', background: 'rgba(245, 158, 11, 0.08)', borderRadius: 8, border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                            <p style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#fbbf24', fontWeight: 600 }}>📚 Topics Covered</p>
                            <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#94a3b8', fontSize: '0.85rem' }}>
                              {selectedAnnouncement.event_data.topics.map((topic: string, idx: number) => (
                                <li key={idx} style={{ marginBottom: '0.2rem' }}>{topic}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Registration Fee & Prizes */}
                        {(selectedAnnouncement.event_data.registration_fee || selectedAnnouncement.event_data.prizes) && (
                          <div style={{ padding: '0.75rem', background: 'rgba(34, 197, 94, 0.08)', borderRadius: 8, border: '1px solid rgba(34, 197, 94, 0.2)' }}>
                            {selectedAnnouncement.event_data.registration_fee && (
                              <p style={{ margin: '0 0 0.3rem', fontSize: '0.85rem', color: '#22c55e' }}>
                                <strong>💰 Registration Fee:</strong> {selectedAnnouncement.event_data.registration_fee}
                              </p>
                            )}
                            {selectedAnnouncement.event_data.prizes && (
                              <p style={{ margin: 0, fontSize: '0.85rem', color: '#22c55e' }}>
                                <strong>🏆 Prizes:</strong> {selectedAnnouncement.event_data.prizes}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Contact Information */}
                        {selectedAnnouncement.event_data.contact_persons && selectedAnnouncement.event_data.contact_persons.length > 0 && (
                          <div style={{ padding: '0.75rem', background: 'rgba(59, 130, 246, 0.08)', borderRadius: 8, border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                            <p style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#60a5fa', fontWeight: 600 }}>📞 Contact Information</p>
                            {selectedAnnouncement.event_data.contact_persons.map((contact: any, idx: number) => (
                              <div key={idx} style={{ marginBottom: '0.5rem', fontSize: '0.85rem', color: '#94a3b8' }}>
                                {typeof contact === 'string' ? (
                                  <p style={{ margin: 0 }}>{contact}</p>
                                ) : (
                                  <>
                                    <p style={{ margin: '0 0 0.2rem', fontWeight: 600, color: '#e2e8f0' }}>
                                      {contact.name} {contact.role && `(${contact.role})`}
                                    </p>
                                    {contact.phone && <p style={{ margin: '0.1rem 0' }}>☎️ {contact.phone}</p>}
                                    {contact.email && <p style={{ margin: '0.1rem 0' }}>✉️ {contact.email}</p>}
                                  </>
                                )}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Additional Information */}
                        {selectedAnnouncement.event_data.additional_info && (
                          <div style={{ padding: '0.75rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 8, border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                            <p style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#e2e8f0', fontWeight: 600 }}>ℹ️ Additional Information</p>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                              {selectedAnnouncement.event_data.additional_info}
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.6 }}>{selectedAnnouncement.body}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
        <div>
          {isCameraOpen && (
            <div className="camera-modal">
              <div className="camera-container">
                <video ref={videoRef} autoPlay playsInline className="camera-video" />
                <div className="camera-controls">
                  <button onClick={capturePhoto} className="capture-btn" disabled={uploading}>
                    {uploading ? 'Saving...' : 'Capture'}
                  </button>
                  <button onClick={closeCamera} className="cancel-btn">Cancel</button>
                </div>
              </div>
            </div>
          )}

          <div className="input-container-dark" style={{
            left: showChatSidebar ? '0px' : '0px',
            right: (sidebarView === 'reminders' || sidebarView === 'announcements') ? '0px' : '0px'
          }}>
            
         <input
      type="file"
      id="file-upload"
      className="file-input-hidden"
      onChange={handleFileUpload}
      disabled={uploading}
      multiple
    />
    <label htmlFor="file-upload" className={`input-icon-btn ${uploading ? 'disabled' : ''}`}>
      <Upload size={20} />
    </label>
    
    <button onClick={openCamera} className="input-icon-btn" disabled={uploading || isCameraOpen}>
      <Camera size={20} />
    </button>
    
    <div style={{ flex: 1, position: 'relative' }}>
     {pendingFiles.length > 0 && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: 0,
          right: 0,
          marginBottom: '0.5rem',
          padding: '0.5rem',
          background: 'rgba(59, 130, 246, 0.15)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: '8px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.5rem'
        }}>
          {pendingFiles.map((file, index) => (
            <div key={index} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'rgba(59, 130, 246, 0.2)',
              borderRadius: '6px',
              padding: '0.3rem 0.5rem',
            }}>
              {pendingFilePreviews[index] ? (
                <img src={pendingFilePreviews[index]} alt="preview" style={{ width: '32px', height: '32px', borderRadius: '4px', objectFit: 'cover' }} />
              ) : (
                <span style={{ fontSize: '1rem' }}>📎</span>
              )}
              <span style={{ fontSize: '0.8rem', color: '#e2e8f0', maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {file.name}
              </span>
              <button onClick={() => removePendingFile(index)} style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: '#ef4444',
                padding: '2px',
                display: 'flex'
              }}>
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
      <input
        type="text"
        value={input}
        onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
        onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleSend()}
        placeholder={uploading ? 'Uploading...' : pendingFiles.length > 0 ? 'Add message for these files...' : 'Type a message...'}
        className="message-input-dark"
        disabled={uploading}
      />
    </div>
    
    <button onClick={handleSend} className="send-btn-dark" disabled={uploading}>
      <Send size={20} />
    </button>
  </div>
   </div>
      

        {/* ── TOAST NOTIFICATIONS ── */}
        <div style={{ position:'fixed', bottom:'5rem', right:'1.5rem', display:'flex', flexDirection:'column', gap:'0.75rem', zIndex:9999, maxWidth: '320px' }}>
          {toasts.map(t => (
            <div key={t.id} style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.6rem',
              padding: '0.75rem 1rem',
              borderRadius: 12,
              fontSize: '0.85rem',
              color: '#fff',
              fontWeight: 500,
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
              background: t.type === 'success' ? '#16a34a' : t.type === 'error' ? '#dc2626' : '#1e40af',
              borderLeft: `4px solid ${t.type === 'success' ? '#4ade80' : t.type === 'error' ? '#f87171' : '#60a5fa'}`,
              animation: 'slideIn 0.3s ease',
              backdropFilter: 'blur(8px)',
            }}>
              <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>
                {t.message.startsWith('⏰') ? '⏰' : t.message.startsWith('📢') ? '📢' : t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : 'ℹ️'}
              </span>
              <span style={{ lineHeight: 1.4 }}>{t.message.replace(/^(⏰|📢|✅|❌)\s*/, '')}</span>
            </div>
          ))}
        </div>
      </div>
  );
}
