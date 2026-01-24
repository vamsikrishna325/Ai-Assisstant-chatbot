import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyBiguICZ25hKkdhbGvLG84OCtXUYHi2MWI",
  authDomain: "chatbot-storage-9d8dc.firebaseapp.com",
  projectId: "chatbot-storage-9d8dc",
  storageBucket: "chatbot-storage-9d8dc.firebasestorage.app",
  messagingSenderId: "694095566692",
  appId: "1:694095566692:web:7ea88e4c0dd9c587b74c71",
  measurementId: "G-69BY31Q9J9"
};

const app = initializeApp(firebaseConfig);
export const analytics = getAnalytics(app);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
export const db = getFirestore(app);
export const storage = getStorage(app);