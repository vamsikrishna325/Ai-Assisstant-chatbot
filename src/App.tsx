import { useState, useEffect, useRef } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { Send, Upload, Camera, Plus, Sparkles, User, Trash2, Share2, MoreVertical, Bell, Megaphone, X, Menu } from 'lucide-react';
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

interface Message {
  text: string;
  sender: string;
  timestamp?: number;
  fileUrl?: string;
  fileType?: string;
  fileName?: string;
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
  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);

  const scrollToBottom = () => {
    if (shouldScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
    if (!input.trim()) return;
    
    shouldScrollRef.current = true;
    
    const newMessage: Message = { 
      text: input, 
      sender: 'user',
      timestamp: Date.now()
    };
    
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setInput('');
    
    const savedChatId = await saveChatToFirestore(updatedMessages, currentChatId);
    if (!currentChatId && savedChatId) {
      setCurrentChatId(savedChatId);
    }

    setTimeout(async () => {
      const botMessage: Message = {
        text: "I'm a AI bot. In a real app, I'd process your message here!",
        sender: 'bot',
        timestamp: Date.now()
      };
      const withBotMessage = [...updatedMessages, botMessage];
      setMessages(withBotMessage);
      await saveChatToFirestore(withBotMessage, currentChatId || savedChatId);
    }, 1000);
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
      alert('File must be less than 10MB');
      e.target.value = '';
      return;
    }

    setUploading(true);
    shouldScrollRef.current = true;

    try {
      const fileUrl = await uploadFileToStorage(file);
      const fileType = file.type.startsWith('image/') ? 'image' : 'file';
      
      const newMessage: Message = { 
        text: `Uploaded: ${file.name}`,
        sender: 'user',
        timestamp: Date.now(),
        fileUrl,
        fileType,
        fileName: file.name
      };
      
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);
      
      const savedChatId = await saveChatToFirestore(updatedMessages, currentChatId);
      if (!currentChatId && savedChatId) {
        setCurrentChatId(savedChatId);
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      alert(`Failed to upload file: ${err.message}`);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const openCamera = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setIsCameraOpen(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Could not access camera');
    }
  };

  const closeCamera = (): void => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
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
      canvas.toBlob(async (blob) => {
        if (blob) {
          setUploading(true);
          try {
            const file = new File([blob], `photo_${Date.now()}.jpg`, { type: 'image/jpeg' });
            const fileUrl = await uploadFileToStorage(file);
            
            const newMessage: Message = { 
              text: 'Photo captured',
              sender: 'user',
              timestamp: Date.now(),
              fileUrl,
              fileType: 'image',
              fileName: file.name
            };
            
            const updatedMessages = [...messages, newMessage];
            setMessages(updatedMessages);
            
            const savedChatId = await saveChatToFirestore(updatedMessages, currentChatId);
            if (!currentChatId && savedChatId) {
              setCurrentChatId(savedChatId);
            }
            
            closeCamera();
          } catch (err: any) {
            console.error('Photo capture error:', err);
            alert(`Failed to save photo: ${err.message}`);
          } finally {
            setUploading(false);
          }
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
            >
              <Bell size={20} />
            </button>
            <button 
              className="header-action-btn"
              onClick={() => setSidebarView(sidebarView === 'announcements' ? 'chat' : 'announcements')}
              title="Announcements"
            >
              <Megaphone size={20} />
            </button>
          </div>
        </div>

        <div className="content-with-sidebar">
          <div className="messages-area-dark">
            {messages.length === 0 ? (
              <div className="welcome-screen">
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
                      <div>{msg.text}</div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Right Sidebar for Reminders/Announcements */}
          {(sidebarView === 'reminders' || sidebarView === 'announcements') && (
            <div className="right-sidebar">
              <div className="right-sidebar-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  {sidebarView === 'reminders' ? <Bell size={20} /> : <Megaphone size={20} />}
                  <h3>{sidebarView === 'reminders' ? 'Reminders' : 'Announcements'}</h3>
                </div>
                <button onClick={() => setSidebarView('chat')} className="close-panel-btn">
                  <X size={18} />
                </button>
              </div>
              <div className="right-sidebar-content">
                <p style={{ color: '#8b92a8', textAlign: 'center', padding: '2rem 1rem' }}>
                  {sidebarView === 'reminders' 
                    ? 'No reminders yet. Create your first reminder!' 
                    : 'No announcements available.'}
                </p>
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

          <div className="input-container-dark">
            <input
              type="file"
              id="file-upload"
              className="file-input-hidden"
              onChange={handleFileUpload}
              disabled={uploading}
            />
            <label htmlFor="file-upload" className={`input-icon-btn ${uploading ? 'disabled' : ''}`}>
              <Upload size={20} />
            </label>
            
            <button onClick={openCamera} className="input-icon-btn" disabled={uploading || isCameraOpen}>
              <Camera size={20} />
            </button>
            
            <input
              type="text"
              value={input}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
              onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleSend()}
              placeholder={uploading ? 'Uploading...' : 'Type a message...'}
              className="message-input-dark"
              disabled={uploading}
            />
            
            <button onClick={handleSend} className="send-btn-dark" disabled={uploading}>
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}