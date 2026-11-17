// Main Dashboard Component
import React, { useState, useEffect, useRef } from 'react';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Home, MessageCircle, TrendingUp, Play, Pause, Volume2, Sun, Moon, Wifi, WifiOff, BarChart3 } from 'lucide-react';


// ======================== UTILITY HOOKS ========================
const useAudioPlayer = () => {
  const [playingAudio, setPlayingAudio] = useState(null);
  const audioRef = useRef(null);

  const playTTSAudio = async (audioData, format, conversationId) => {
    try {
      if (playingAudio === conversationId) {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }
        setPlayingAudio(null);
        return;
      }

      let audioUrl;
      if (format === 'base64') {
        const binaryString = atob(audioData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const audioBlob = new Blob([bytes], { type: 'audio/wav' });
        audioUrl = URL.createObjectURL(audioBlob);
      } else if (format === 'url') {
        audioUrl = `/generate`;
      }

      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      setPlayingAudio(conversationId);

      audio.onended = () => {
        setPlayingAudio(null);
        if (audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(audioUrl);
        }
      };

      audio.onerror = () => {
        setPlayingAudio(null);
        if (audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(audioUrl);
        }
      };

      await audio.play();
    } catch (error) {
      console.error('Audio playback failed:', error);
      setPlayingAudio(null);
    }
  };

  return { playingAudio, playTTSAudio };
};

const useNotifications = () => {
  const [notifications, setNotifications] = useState([]);

  const showNotification = (message, type = 'info') => {
    const notification = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toISOString()
    };
    
    setNotifications(prev => [notification, ...prev].slice(0, 5));
    
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== notification.id));
    }, 4000);
  };

  return { notifications, showNotification };
};

// ======================== UI COMPONENTS ========================
const CircularProgress = ({ value, max = 10, darkMode }) => {
  const percentage = (value / max) * 100;
  const strokeDasharray = 2 * Math.PI * 40;
  const strokeDashoffset = strokeDasharray - (strokeDasharray * percentage) / 100;
  
  return (
    <div className="relative w-32 h-32">
      <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
        <circle
          cx="50"
          cy="50"
          r="40"
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          className="text-gray-200 dark:text-gray-700"
        />
        <circle
          cx="50"
          cy="50"
          r="40"
          stroke="currentColor"
          strokeWidth="8"
          fill="none"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          className="text-blue-500 transition-all duration-500 ease-in-out"
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-2xl font-bold text-gray-800 dark:text-white">
          {value.toFixed(1)}
        </span>
      </div>
    </div>
  );
};

const NotificationBar = ({ notifications }) => (
  notifications.length > 0 && (
    <div className="fixed top-20 right-4 z-50 space-y-2">
      {notifications.map(notification => (
        <div
          key={notification.id}
          className={`px-4 py-2 rounded-lg shadow-lg text-white text-sm animate-slide-in ${
            notification.type === 'error' ? 'bg-red-500' :
            notification.type === 'success' ? 'bg-green-500' :
            'bg-blue-500'
          }`}
        >
          {notification.message}
        </div>
      ))}
    </div>
  )
);

const Navigation = ({ activeTab, setActiveTab, darkMode, toggleTheme, connectionStatus }) => (
  <nav className={`${darkMode ? 'bg-gray-900' : 'bg-white'} shadow-lg transition-colors duration-200`}>
    <div className="flex items-center justify-between p-4">
      <h1 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        Mood Tracker Dashboard
      </h1>
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`flex items-center px-4 py-2 rounded-lg transition-all ${
            activeTab === 'dashboard' 
              ? 'bg-blue-600 text-white shadow-lg' 
              : `${darkMode ? 'text-gray-300 hover:bg-gray-800' : 'text-gray-600 hover:bg-gray-100'}`
          }`}
        >
          <Home size={18} className="mr-2" />
          Dashboard
        </button>
        <button
          onClick={() => setActiveTab('insights')}
          className={`flex items-center px-4 py-2 rounded-lg transition-all ${
            activeTab === 'insights' 
              ? 'bg-blue-600 text-white shadow-lg' 
              : `${darkMode ? 'text-gray-300 hover:bg-gray-800' : 'text-gray-600 hover:bg-gray-100'}`
          }`}
        >
          <TrendingUp size={18} className="mr-2" />
          Insights
        </button>
        <button
          onClick={() => setActiveTab('conversations')}
          className={`flex items-center px-4 py-2 rounded-lg transition-all ${
            activeTab === 'conversations' 
              ? 'bg-blue-600 text-white shadow-lg' 
              : `${darkMode ? 'text-gray-300 hover:bg-gray-800' : 'text-gray-600 hover:bg-gray-100'}`
          }`}
        >
          <MessageCircle size={18} className="mr-2" />
          Conversations
        </button>
      </div>
      <div className="flex items-center space-x-3">
        <button
          onClick={toggleTheme}
          className={`p-2 rounded-lg transition-colors ${
            darkMode ? 'bg-gray-800 text-yellow-400' : 'bg-gray-100 text-gray-600'
          }`}
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
          connectionStatus === 'connected' 
            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
            : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
        }`}>
          {connectionStatus === 'connected' ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span className="ml-1 capitalize">{connectionStatus}</span>
        </div>
      </div>
    </div>
  </nav>
);

// ======================== CARD COMPONENTS ========================
const MoodScoreCard = ({ currentMood, darkMode }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      Current Mood Score
    </h3>
    <div className="flex items-center justify-center">
      <CircularProgress value={currentMood} max={10} darkMode={darkMode} />
    </div>
    <p className={`text-center mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
      Out of 10 - {currentMood >= 7 ? 'Great!' : currentMood >= 5 ? 'Good' : currentMood >= 3 ? 'Okay' : 'Needs attention'}
    </p>
  </div>
);

const MessageInputCard = ({ darkMode, sendMessageToPi }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      Talk to Mood Bot
    </h3>
    <div className="space-y-3">
      <textarea
        id="moodMessage"
        placeholder="How are you feeling today? Share your thoughts..."
        className={`w-full p-3 rounded-lg border transition-colors duration-200 ${
          darkMode 
            ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
            : 'bg-white border-gray-300 text-gray-800'
        } focus:ring-2 focus:ring-blue-500 focus:border-transparent`}
        rows="4"
      />
      <button
        onClick={() => {
          const textarea = document.getElementById('moodMessage');
          if (textarea.value.trim()) {
            sendMessageToPi(textarea.value.trim());
            textarea.value = '';
          }
        }}
        className="w-full px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg"
      >
        Send Message
      </button>
    </div>
  </div>
);

const MoodTrendsChart = ({ moodTrends, darkMode }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      Weekly Mood Trends
    </h3>
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={moodTrends}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
        <XAxis dataKey="day" className={darkMode ? 'text-gray-300' : 'text-gray-600'} />
        <YAxis domain={[0, 10]} className={darkMode ? 'text-gray-300' : 'text-gray-600'} />
        <Tooltip 
          contentStyle={{
            backgroundColor: darkMode ? '#374151' : '#ffffff',
            border: 'none',
            borderRadius: '8px',
            color: darkMode ? '#ffffff' : '#000000'
          }}
        />
        <Area
          type="monotone"
          dataKey="mood"
          stroke="#3b82f6"
          fill="#3b82f6"
          fillOpacity={0.3}
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  </div>
);

const ConversationCountsChart = ({ conversationCounts, darkMode }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      Daily Conversations
    </h3>
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={conversationCounts}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
        <XAxis dataKey="day" className={darkMode ? 'text-gray-300' : 'text-gray-600'} />
        <YAxis className={darkMode ? 'text-gray-300' : 'text-gray-600'} />
        <Tooltip 
          contentStyle={{
            backgroundColor: darkMode ? '#374151' : '#ffffff',
            border: 'none',
            borderRadius: '8px',
            color: darkMode ? '#ffffff' : '#000000'
          }}
        />
        <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

const RecentBotResponses = ({ conversations, darkMode, playingAudio, playTTSAudio }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <div className="flex items-center mb-6">
      <Volume2 className="mr-3 text-indigo-500" size={24} />
      <h3 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        Recent Bot Responses - Click Play to Listen
      </h3>
    </div>
    <div className="space-y-4 max-h-80 overflow-y-auto">
      {conversations.filter(conv => conv.type === 'bot' && conv.audio_data).slice(0, 5).map(conv => (
        <div key={conv.id} className={`border-l-4 border-indigo-500 p-4 rounded-r-lg transition-colors duration-200 ${
          darkMode ? 'bg-indigo-900/20' : 'bg-indigo-50'
        }`}>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className={`${darkMode ? 'text-gray-200' : 'text-gray-800'} mb-2`}>
                {conv.message}
              </p>
              <div className="flex items-center space-x-4">
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {new Date(conv.timestamp).toLocaleString()}
                </p>
                {conv.mood_score && (
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    conv.mood_score >= 7 ? 'bg-green-200 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    conv.mood_score >= 4 ? 'bg-yellow-200 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    'bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    Mood: {conv.mood_score.toFixed(1)}
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={() => playTTSAudio(conv.audio_data, conv.audio_format || 'base64', conv.id)}
              className={`ml-4 px-4 py-2 rounded-full flex items-center transition-all shadow-lg hover:shadow-xl ${
                playingAudio === conv.id 
                  ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' 
                  : 'bg-indigo-500 hover:bg-indigo-600 text-white hover:scale-105'
              }`}
            >
              {playingAudio === conv.id ? <Pause size={18} /> : <Play size={18} />}
              <span className="ml-2 font-medium">
                {playingAudio === conv.id ? 'Stop Audio' : 'Play TTS'}
              </span>
            </button>
          </div>
        </div>
      ))}
      {conversations.filter(conv => conv.type === 'bot' && conv.audio_data).length === 0 && (
        <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <Volume2 size={48} className="mx-auto mb-4 opacity-50" />
          <p>No audio responses available yet.</p>
          <p className="text-sm">Send a message to the mood bot to get TTS responses!</p>
        </div>
      )}
    </div>
  </div>
);

const InsightCard = ({ icon: Icon, title, value, color, darkMode }) => (
  <div className={`p-6 rounded-xl shadow-lg transition-colors duration-200 ${
    darkMode ? 'bg-gray-800' : 'bg-white'
  }`}>
    <div className="flex items-center">
      <Icon className={`mr-3 ${color}`} size={24} />
      <div>
        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{title}</p>
        <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
          {typeof value === 'number' ? value.toFixed(1) : value}
        </p>
      </div>
    </div>
  </div>
);

// ======================== PAGE COMPONENTS ========================
const DashboardPage = ({ 
  currentMood, 
  moodTrends, 
  conversationCounts, 
  conversations, 
  darkMode, 
  sendMessageToPi, 
  playingAudio, 
  playTTSAudio 
}) => (
  <div className={`p-6 min-h-screen transition-colors duration-200 ${
    darkMode ? 'bg-gray-900' : 'bg-gray-50'
  }`}>
    {/* Current Mood and Message Input */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <MoodScoreCard currentMood={currentMood} darkMode={darkMode} />
      <MessageInputCard darkMode={darkMode} sendMessageToPi={sendMessageToPi} />
    </div>

    {/* Charts */}
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
      <MoodTrendsChart moodTrends={moodTrends} darkMode={darkMode} />
      <ConversationCountsChart conversationCounts={conversationCounts} darkMode={darkMode} />
    </div>

    {/* Recent Bot Responses with TTS Audio */}
    <RecentBotResponses 
      conversations={conversations} 
      darkMode={darkMode} 
      playingAudio={playingAudio} 
      playTTSAudio={playTTSAudio} 
    />
  </div>
);

const InsightsPage = ({ moodInsights, darkMode }) => (
  <div className={`p-6 min-h-screen transition-colors duration-200 ${
    darkMode ? 'bg-gray-900' : 'bg-gray-50'
  }`}>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <InsightCard 
        icon={BarChart3} 
        title="Average Mood" 
        value={moodInsights.averageMood} 
        color="text-blue-500" 
        darkMode={darkMode} 
      />
      <InsightCard 
        icon={TrendingUp} 
        title="Best Day" 
        value={moodInsights.bestDay} 
        color="text-green-500" 
        darkMode={darkMode} 
      />
      <InsightCard 
        icon={MessageCircle} 
        title="Total Conversations" 
        value={moodInsights.totalConversations} 
        color="text-purple-500" 
        darkMode={darkMode} 
      />
      <InsightCard 
        icon={Volume2} 
        title="Mood Stability" 
        value={`${moodInsights.moodStability}%`} 
        color="text-indigo-500" 
        darkMode={darkMode} 
      />
    </div>
  </div>
);

const ConversationsPage = ({ conversations, darkMode, playingAudio, playTTSAudio }) => (
  <div className={`p-6 min-h-screen transition-colors duration-200 ${
    darkMode ? 'bg-gray-900' : 'bg-gray-50'
  }`}>
    <div className={`rounded-xl shadow-lg transition-colors duration-200 ${
      darkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="p-6">
        <h3 className={`text-xl font-semibold mb-6 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
          Conversation History
        </h3>
        
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {conversations.map(conv => (
            <div key={conv.id} className={`border-l-4 p-4 rounded-r-lg transition-colors duration-200 ${
              conv.type === 'user' 
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                : 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className={`${darkMode ? 'text-gray-200' : 'text-gray-800'} mb-2`}>
                    {conv.message}
                  </p>
                  <div className="flex items-center space-x-4">
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      {new Date(conv.timestamp).toLocaleString()}
                    </p>
                    {conv.mood_score && (
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        conv.mood_score >= 7 ? 'bg-green-200 text-green-800 dark:bg-green-900 dark:text-green-200' :
                        conv.mood_score >= 4 ? 'bg-yellow-200 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        'bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        Mood: {conv.mood_score.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
                {conv.audio_data && (
                  <button
                    onClick={() => playTTSAudio(conv.audio_data, conv.audio_format || 'base64', conv.id)}
                    className={`ml-4 px-3 py-2 rounded-full flex items-center transition-all ${
                      playingAudio === conv.id 
                        ? 'bg-red-500 hover:bg-red-600 text-white' 
                        : 'bg-purple-500 hover:bg-purple-600 text-white'
                    }`}
                  >
                    {playingAudio === conv.id ? <Pause size={16} /> : <Play size={16} />}
                    <span className="ml-2">{playingAudio === conv.id ? 'Stop' : 'Play'}</span>
                  </button>
                )}
              </div>
            </div>
          ))}
          {conversations.length === 0 && (
            <p className={`text-center py-12 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              No conversations yet. Start chatting with the mood bot!
            </p>
          )}
        </div>
      </div>
    </div>
  </div>
);

// ======================== MAIN APP COMPONENT ========================
const Dashboard = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [darkMode, setDarkMode] = useState(false);
  const [currentMood, setCurrentMood] = useState(7.2);
  const [moodTrends, setMoodTrends] = useState([]);
  const [conversationCounts, setConversationCounts] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [moodInsights, setMoodInsights] = useState({
    averageMood: 0,
    bestDay: 'Monday',
    totalConversations: 0,
    moodStability: 0
  });
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastConnectionTime, setLastConnectionTime] = useState(null);

  // Custom hooks
  const { playingAudio, playTTSAudio } = useAudioPlayer();
  const { notifications, showNotification } = useNotifications();

  const PI_BASE_URL = '/generate';

  // API functions
  const fetchMoodData = async () => {
    try {
      const response = await fetch(`${PI_BASE_URL}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      
      if (data.current_mood !== undefined) setCurrentMood(data.current_mood);
      if (data.mood_trends) setMoodTrends(data.mood_trends);
      if (data.conversation_counts) setConversationCounts(data.conversation_counts);
      if (data.insights) setMoodInsights(data.insights);

      if (data.conversations) {
        setConversations(prev => {
          const newConversations = data.conversations.filter(conv => 
            !prev.some(existing => existing.id === conv.id)
          );
          
          if (newConversations.length > 0) {
            showNotification(`${newConversations.length} new conversation(s)`, 'info');
          }
          
          return [...newConversations, ...prev].slice(0, 100);
        });
      }

      setConnectionStatus('connected');
      setLastConnectionTime(new Date().toISOString());
      
    } catch (error) {
      console.error('Failed to fetch mood data from Pi:', error);
      setConnectionStatus('disconnected');
      showNotification('Connection to Pi lost', 'error');
    }
  };

  const sendMessageToPi = async (message) => {
    try {
      const response = await fetch(`${PI_BASE_URL}/api/mood-message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: message,
          timestamp: new Date().toISOString()
        })
      });

      if (response.ok) {
        showNotification('Message sent to mood bot', 'success');
        setTimeout(fetchMoodData, 1000);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Failed to send message to Pi:', error);
      showNotification('Failed to send message', 'error');
    }
  };

  const toggleTheme = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  // Effects
  useEffect(() => {
    fetchMoodData();
    const interval = setInterval(fetchMoodData, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);
  

  // Render
  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-100'
    }`}>
      <Navigation 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        darkMode={darkMode} 
        toggleTheme={toggleTheme} 
        connectionStatus={connectionStatus} 
      />
      <NotificationBar notifications={notifications} />
      
      {activeTab === 'dashboard' && (
        <DashboardPage 
          currentMood={currentMood}
          moodTrends={moodTrends}
          conversationCounts={conversationCounts}
          conversations={conversations}
          darkMode={darkMode}
          sendMessageToPi={sendMessageToPi}
          playingAudio={playingAudio}
          playTTSAudio={playTTSAudio}
        />
      )}
      {activeTab === 'insights' && (
        <InsightsPage moodInsights={moodInsights} darkMode={darkMode} />
      )}
      {activeTab === 'conversations' && (
        <ConversationsPage 
          conversations={conversations}
          darkMode={darkMode}
          playingAudio={playingAudio}
          playTTSAudio={playTTSAudio}
        />
      )}

      <style jsx>{`
        @keyframes slide-in {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
        
        .dark {
          color-scheme: dark;
        }
      `}</style>
    </div>
  );
};

export default Dashboard;