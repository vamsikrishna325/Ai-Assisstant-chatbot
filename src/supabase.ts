import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://ionrcywccbyivttkxdxc.supabase.co'; 
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlvbnJjeXdjY2J5aXZ0dGt4ZHhjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTI0MjI3MSwiZXhwIjoyMDg0ODE4MjcxfQ.G8_5A4XcwxoEGcQ3frsnJ4yXAIqT5qsJ39rXS9HbaYM';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: false
  }
});