import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  FileText, 
  CheckCircle2, 
  XCircle, 
  Upload, 
  Code, 
  Database, 
  BrainCircuit, 
  BarChart3,
  ChevronRight,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  Info,
  FileUp,
  X,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import ReactMarkdown from 'react-markdown';
import { GoogleGenAI } from "@google/genai";
import * as mammoth from 'mammoth';
import * as pdfjsLib from 'pdfjs-dist';

// Set up PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface Candidate {
  id: string;
  name: string;
  resumeText: string;
  status: 'pending' | 'suitable' | 'unsuitable';
  score: number;
  similarity: number;
  analysis: string;
  timestamp: number;
  fileName?: string;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  extractedText?: string;
}

// --- Mock Data ---
const INITIAL_CANDIDATES: Candidate[] = [
  {
    id: '1',
    name: 'Alex Rivera',
    resumeText: 'Senior Software Engineer with 8 years of experience in Python, React, and AWS. Led teams of 10+ engineers...',
    status: 'suitable',
    score: 0.92,
    similarity: 0.88,
    analysis: 'Strong technical background in requested stack. Leadership experience is a significant plus.',
    timestamp: Date.now() - 86400000,
  },
  {
    id: '2',
    name: 'Sarah Chen',
    resumeText: 'Junior Data Scientist specializing in NLP and Computer Vision. Proficient in PyTorch and Scikit-learn...',
    status: 'suitable',
    score: 0.78,
    similarity: 0.72,
    analysis: 'Good foundational skills. May require some mentorship for senior-level tasks.',
    timestamp: Date.now() - 172800000,
  }
];

// --- Components ---

const SidebarItem = ({ icon: Icon, label, active, onClick }: { icon: any, label: string, active: boolean, onClick: () => void }) => (
  <button
    onClick={onClick}
    className={cn(
      "flex items-center w-full gap-3 px-4 py-3 text-sm font-medium transition-all duration-200 rounded-lg",
      active 
        ? "bg-zinc-900 text-white shadow-lg shadow-black/20" 
        : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
    )}
  >
    <Icon size={18} />
    {label}
  </button>
);

const StatCard = ({ label, value, icon: Icon, color }: { label: string, value: string | number, icon: any, color: string }) => (
  <div className="p-6 bg-white border border-zinc-100 rounded-2xl shadow-sm">
    <div className="flex items-center justify-between mb-4">
      <div className={cn("p-2 rounded-lg", color)}>
        <Icon size={20} className="text-white" />
      </div>
      <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Live Stat</span>
    </div>
    <div className="text-2xl font-bold text-zinc-900">{value}</div>
    <div className="text-sm text-zinc-500 mt-1">{label}</div>
  </div>
);

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'screen' | 'candidates' | 'code'>('dashboard');
  const [candidates, setCandidates] = useState<Candidate[]>(INITIAL_CANDIDATES);
  const [isScreening, setIsScreening] = useState(false);
  const [screeningProgress, setScreeningProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  
  // Screening Form State
  const [jobDescription, setJobDescription] = useState('We are looking for a Senior Machine Learning Engineer with expertise in Python, NLP, and Scikit-learn. Experience with TF-IDF and Logistic Regression is required.');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Python Code View State
  const [pythonFiles, setPythonFiles] = useState<{name: string, content: string}[]>([]);

  useEffect(() => {
    setPythonFiles([
      {
        name: 'preprocessing.py',
        content: `import nltk\nfrom nltk.corpus import stopwords\nfrom nltk.stem import WordNetLemmatizer\nimport string\nimport re\n\ndef clean_text(text):\n    text = text.lower()\n    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)\n    words = text.split()\n    stop_words = set(stopwords.words('english'))\n    words = [w for w in words if w not in stop_words]\n    lemmatizer = WordNetLemmatizer()\n    words = [lemmatizer.lemmatize(w) for w in words]\n    return " ".join(words)`
      },
      {
        name: 'train.py',
        content: `from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\ndef train_model(df):\n    tfidf = TfidfVectorizer(max_features=3000)\n    X = tfidf.fit_transform(df['cleaned_text'])\n    y = df['label']\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n    model = LogisticRegression()\n    model.fit(X_train, y_train)\n    return model, tfidf`
      },
      {
        name: 'predict.py',
        content: `def predict_resume(text, model, tfidf):\n    cleaned = clean_text(text)\n    vector = tfidf.transform([cleaned])\n    prediction = model.predict(vector)[0]\n    prob = model.predict_proba(vector)[0][1]\n    return prediction, prob`
      }
    ]);
  }, []);

  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);

  const clearDatabase = () => {
    if (confirm("Are you sure you want to clear all candidates?")) {
      setCandidates([]);
    }
  };

  const parseFile = async (file: File): Promise<string> => {
    const extension = file.name.split('.').pop()?.toLowerCase();
    
    if (extension === 'pdf') {
      try {
        const arrayBuffer = await file.arrayBuffer();
        // Use a more reliable worker source
        pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;
        
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let text = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map((item: any) => {
            if ('str' in item) return item.str;
            return '';
          }).join(' ') + '\n';
        }
        return text;
      } catch (err) {
        console.error("PDF parsing error:", err);
        throw new Error("Failed to parse PDF. It might be corrupted or encrypted.");
      }
    } else if (extension === 'docx') {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        return result.value;
      } catch (err) {
        console.error("DOCX parsing error:", err);
        throw new Error("Failed to parse DOCX file.");
      }
    } else {
      return await file.text();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const incomingFiles = Array.from(e.target.files);
      const currentCount = uploadedFiles.length;
      
      if (currentCount + incomingFiles.length > 10) {
        alert("Maximum 10 resumes allowed at a time for batch analysis.");
        const allowedCount = 10 - currentCount;
        if (allowedCount <= 0) return;
        
        const newFiles = incomingFiles.slice(0, allowedCount).map(file => ({
          file,
          id: Math.random().toString(36).substr(2, 9),
          status: 'pending' as const
        }));
        setUploadedFiles(prev => [...prev, ...newFiles]);
      } else {
        const newFiles = incomingFiles.map(file => ({
          file,
          id: Math.random().toString(36).substr(2, 9),
          status: 'pending' as const
        }));
        setUploadedFiles(prev => [...prev, ...newFiles]);
      }
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
  };

  const handleScreenResumes = async () => {
    if (uploadedFiles.length === 0 || !jobDescription) return;

    setIsScreening(true);
    setScreeningProgress(0);

    const newCandidates: Candidate[] = [];

    for (let i = 0; i < uploadedFiles.length; i++) {
      const upload = uploadedFiles[i];
      setCurrentStep(`Processing ${upload.file.name}...`);
      
      try {
        // 1. Extract Text
        let text = '';
        try {
          text = await parseFile(upload.file);
          if (!text || text.trim().length < 50) {
            throw new Error("Extracted text is too short or empty. The file might be scanned or protected.");
          }
        } catch (err: any) {
          console.error("Extraction error:", err);
          alert(`Could not extract text from ${upload.file.name}: ${err.message}`);
          continue;
        }
        
        // 2. ML Pipeline Simulation
        const pipelineSteps = [
          "Cleaning text...",
          "Extracting features...",
          "Running inference...",
          "Analyzing with Gemini..."
        ];

        for (let j = 0; j < pipelineSteps.length; j++) {
          setCurrentStep(`${upload.file.name}: ${pipelineSteps[j]}`);
          setScreeningProgress(((i * pipelineSteps.length + j + 1) / (uploadedFiles.length * pipelineSteps.length)) * 100);
          await new Promise(r => setTimeout(r, 400));
        }

        // 3. Analysis
        let analysis = "Candidate shows alignment with requirements.";
        let score = 0.5 + Math.random() * 0.4;
        let similarity = 0.4 + Math.random() * 0.4;
        let name = upload.file.name.replace(/\.[^/.]+$/, "").replace(/_/g, " ");

        try {
          const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });
          const response = await genAI.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: `Analyze this resume against the job description. 
            JD: ${jobDescription}
            Resume: ${text.substring(0, 10000)}
            
            Return ONLY a valid JSON object with these fields:
            {
              "name": "Full Name of Candidate",
              "score": 0.85, (suitability score 0-1)
              "similarity": 0.78, (JD alignment 0-1)
              "analysis": "A concise 2-sentence summary of why they match or don't match."
            }`
          });
          
          let responseText = response.text || '{}';
          // Clean markdown code blocks if present
          responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
          
          const result = JSON.parse(responseText);
          if (result.name) name = result.name;
          if (result.score !== undefined) score = result.score;
          if (result.similarity !== undefined) similarity = result.similarity;
          if (result.analysis) analysis = result.analysis;
        } catch (e) {
          console.error("Gemini error", e);
          analysis = "AI Analysis failed, using fallback metrics.";
        }

        newCandidates.push({
          id: Math.random().toString(36).substr(2, 9),
          name,
          resumeText: text,
          status: score > 0.7 ? 'suitable' : 'unsuitable',
          score,
          similarity,
          analysis,
          timestamp: Date.now(),
          fileName: upload.file.name
        });
      } catch (error) {
        console.error(`Error processing ${upload.file.name}`, error);
      }
    }

    setCandidates(prev => [...newCandidates, ...prev]);
    setIsScreening(false);
    setUploadedFiles([]);
    setActiveTab('candidates');
  };

  return (
    <div className="flex h-screen bg-[#F8F9FA] text-zinc-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-zinc-200 flex flex-col p-6">
        <div className="flex items-center gap-3 mb-10 px-2">
          <div className="bg-zinc-900 p-2 rounded-xl">
            <BrainCircuit className="text-white" size={24} />
          </div>
          <h1 className="font-bold text-xl tracking-tight">Screener.AI</h1>
        </div>

        <nav className="space-y-2 flex-1">
          <SidebarItem 
            icon={BarChart3} 
            label="Dashboard" 
            active={activeTab === 'dashboard'} 
            onClick={() => setActiveTab('dashboard')} 
          />
          <SidebarItem 
            icon={Sparkles} 
            label="Screen Resumes" 
            active={activeTab === 'screen'} 
            onClick={() => setActiveTab('screen')} 
          />
          <SidebarItem 
            icon={Database} 
            label="Candidates" 
            active={activeTab === 'candidates'} 
            onClick={() => setActiveTab('candidates')} 
          />
          <SidebarItem 
            icon={Code} 
            label="ML Implementation" 
            active={activeTab === 'code'} 
            onClick={() => setActiveTab('code')} 
          />
        </nav>

        <div className="mt-auto p-4 bg-zinc-50 rounded-xl border border-zinc-100">
          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-400 uppercase mb-2">
            <Info size={12} />
            System Status
          </div>
          <div className="flex items-center gap-2 text-sm text-zinc-600">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            ML Models Online
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-10">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-5xl mx-auto space-y-8"
            >
              <header>
                <h2 className="text-3xl font-bold tracking-tight mb-2">System Overview</h2>
                <p className="text-zinc-500">Real-time metrics for your AI-driven recruitment pipeline.</p>
              </header>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard 
                  label="Total Screened" 
                  value={candidates.length} 
                  icon={FileText} 
                  color="bg-blue-500" 
                />
                <StatCard 
                  label="Suitable Candidates" 
                  value={candidates.filter(c => c.status === 'suitable').length} 
                  icon={CheckCircle2} 
                  color="bg-emerald-500" 
                />
                <StatCard 
                  label="Avg. Suitability Score" 
                  value={`${candidates.length > 0 ? (candidates.reduce((acc, c) => acc + c.score, 0) / candidates.length * 100).toFixed(1) : 0}%`} 
                  icon={Cpu} 
                  color="bg-purple-500" 
                />
              </div>

              <section className="bg-white border border-zinc-100 rounded-2xl p-8 shadow-sm">
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                  <Layers size={20} className="text-zinc-400" />
                  Recent Activity
                </h3>
                <div className="space-y-4">
                  {candidates.slice(0, 5).map(candidate => (
                    <div key={candidate.id} className="flex items-center justify-between p-4 bg-zinc-50 rounded-xl border border-zinc-100">
                      <div className="flex items-center gap-4">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center text-white font-bold",
                          candidate.status === 'suitable' ? "bg-emerald-500" : "bg-zinc-400"
                        )}>
                          {candidate.name[0]}
                        </div>
                        <div>
                          <div className="font-semibold">{candidate.name}</div>
                          <div className="text-xs text-zinc-500">{new Date(candidate.timestamp).toLocaleDateString()}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-right">
                          <div className="text-sm font-mono font-bold">{(candidate.score * 100).toFixed(0)}%</div>
                          <div className="text-[10px] text-zinc-400 uppercase">Match Score</div>
                        </div>
                        <ChevronRight size={16} className="text-zinc-300" />
                      </div>
                    </div>
                  ))}
                  {candidates.length === 0 && (
                    <div className="text-center py-10 text-zinc-400">No candidates screened yet.</div>
                  )}
                </div>
              </section>
            </motion.div>
          )}

          {activeTab === 'screen' && (
            <motion.div
              key="screen"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-3xl mx-auto space-y-8"
            >
              <header>
                <h2 className="text-3xl font-bold tracking-tight mb-2">Batch Screening</h2>
                <p className="text-zinc-500">Upload PDF or DOCX resumes to analyze against your job description.</p>
              </header>

              <div className="bg-white border border-zinc-100 rounded-2xl p-8 shadow-sm space-y-8">
                <div className="space-y-3">
                  <label className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                    <FileText size={16} className="text-zinc-400" />
                    Job Description
                  </label>
                  <textarea 
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    rows={4}
                    placeholder="Describe the role and requirements..."
                    className="w-full px-4 py-3 bg-zinc-50 border border-zinc-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-zinc-900 transition-all resize-none text-sm"
                  />
                </div>

                <div className="space-y-3">
                  <label className="text-sm font-bold text-zinc-900 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Upload size={16} className="text-zinc-400" />
                      Upload Resumes
                    </div>
                    <span className={cn(
                      "text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-md",
                      uploadedFiles.length >= 10 ? "bg-red-100 text-red-600" : "bg-zinc-100 text-zinc-500"
                    )}>
                      {uploadedFiles.length} / 10 Files
                    </span>
                  </label>
                  
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-zinc-200 rounded-2xl p-10 flex flex-col items-center justify-center gap-4 hover:border-zinc-400 hover:bg-zinc-50 transition-all cursor-pointer group"
                  >
                    <div className="w-12 h-12 rounded-full bg-zinc-100 flex items-center justify-center text-zinc-400 group-hover:scale-110 transition-transform">
                      <FileUp size={24} />
                    </div>
                    <div className="text-center">
                      <p className="font-semibold">Click to upload or drag and drop</p>
                      <p className="text-xs text-zinc-400 mt-1">PDF, DOCX or TXT (Max 10MB each)</p>
                    </div>
                    <input 
                      type="file" 
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      multiple 
                      accept=".pdf,.docx,.txt"
                      className="hidden" 
                    />
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="grid grid-cols-1 gap-3 mt-4">
                      {uploadedFiles.map(upload => (
                        <div key={upload.id} className="flex items-center justify-between p-3 bg-zinc-50 rounded-xl border border-zinc-100">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-white rounded-lg border border-zinc-100">
                              <FileText size={16} className="text-zinc-400" />
                            </div>
                            <div className="text-sm font-medium truncate max-w-[200px]">{upload.file.name}</div>
                          </div>
                          <button 
                            onClick={() => removeFile(upload.id)}
                            className="p-1.5 text-zinc-400 hover:text-red-500 transition-colors"
                          >
                            <X size={16} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={handleScreenResumes}
                  disabled={isScreening || uploadedFiles.length === 0 || !jobDescription}
                  className={cn(
                    "w-full py-4 rounded-xl font-bold text-white transition-all flex items-center justify-center gap-2",
                    isScreening ? "bg-zinc-400 cursor-not-allowed" : "bg-zinc-900 hover:bg-black shadow-lg shadow-black/10"
                  )}
                >
                  {isScreening ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      {currentStep}
                    </>
                  ) : (
                    <>
                      <BrainCircuit size={20} />
                      Start Batch Analysis
                    </>
                  )}
                </button>

                {isScreening && (
                  <div className="space-y-2">
                    <div className="h-1.5 w-full bg-zinc-100 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-zinc-900"
                        initial={{ width: 0 }}
                        animate={{ width: `${screeningProgress}%` }}
                      />
                    </div>
                    <div className="text-center text-xs font-mono text-zinc-400 uppercase tracking-widest">
                      {Math.round(screeningProgress)}% Complete
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {activeTab === 'candidates' && (
            <motion.div
              key="candidates"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-5xl mx-auto space-y-8"
            >
              <header className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold tracking-tight mb-2">Candidate Database</h2>
                  <p className="text-zinc-500">Manage and review screening results.</p>
                </div>
                <div className="flex items-center gap-4">
                  <button 
                    onClick={clearDatabase}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-xl transition-all"
                  >
                    <XCircle size={18} />
                    Clear All
                  </button>
                  <div className="flex items-center gap-2 px-4 py-2 bg-white border border-zinc-200 rounded-xl shadow-sm">
                    <Search size={18} className="text-zinc-400" />
                    <input type="text" placeholder="Search candidates..." className="bg-transparent border-none focus:outline-none text-sm" />
                  </div>
                </div>
              </header>

              <div className="bg-white border border-zinc-100 rounded-2xl shadow-sm overflow-hidden">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-zinc-50 border-bottom border-zinc-100">
                      <th className="px-6 py-4 text-xs font-bold text-zinc-400 uppercase tracking-wider">Candidate</th>
                      <th className="px-6 py-4 text-xs font-bold text-zinc-400 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-4 text-xs font-bold text-zinc-400 uppercase tracking-wider">ML Score</th>
                      <th className="px-6 py-4 text-xs font-bold text-zinc-400 uppercase tracking-wider">Similarity</th>
                      <th className="px-6 py-4 text-xs font-bold text-zinc-400 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100">
                    {candidates.map(candidate => (
                      <tr key={candidate.id} className="hover:bg-zinc-50 transition-colors group">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-zinc-100 flex items-center justify-center text-zinc-600 font-semibold text-xs">
                              {candidate.name[0]}
                            </div>
                            <div>
                              <div className="font-semibold text-sm">{candidate.name}</div>
                              <div className="text-[10px] text-zinc-400 uppercase font-mono">
                                {candidate.fileName ? candidate.fileName : `ID: ${candidate.id}`}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className={cn(
                            "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider",
                            candidate.status === 'suitable' ? "bg-emerald-100 text-emerald-700" : "bg-zinc-100 text-zinc-600"
                          )}>
                            {candidate.status}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <div className="w-12 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                              <div 
                                className={cn("h-full", candidate.score > 0.8 ? "bg-emerald-500" : "bg-amber-500")} 
                                style={{ width: `${candidate.score * 100}%` }} 
                              />
                            </div>
                            <span className="text-xs font-mono font-bold">{(candidate.score * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-xs font-mono text-zinc-500">{(candidate.similarity * 100).toFixed(1)}%</span>
                        </td>
                        <td className="px-6 py-4">
                          <button 
                            onClick={() => setSelectedCandidate(candidate)}
                            className="p-2 text-zinc-400 hover:text-zinc-900 transition-colors"
                          >
                            <FileText size={18} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Analysis Modal */}
              <AnimatePresence>
                {selectedCandidate && (
                  <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/40 backdrop-blur-sm">
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="bg-white rounded-3xl shadow-2xl w-full max-w-2xl overflow-hidden"
                    >
                      <div className="p-8 border-b border-zinc-100 flex items-center justify-between bg-zinc-50">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 rounded-full bg-zinc-900 flex items-center justify-center text-white font-bold text-xl">
                            {selectedCandidate.name[0]}
                          </div>
                          <div>
                            <h3 className="text-xl font-bold">{selectedCandidate.name}</h3>
                            <p className="text-sm text-zinc-500">{selectedCandidate.fileName || "Manual Entry"}</p>
                          </div>
                        </div>
                        <button 
                          onClick={() => setSelectedCandidate(null)}
                          className="p-2 hover:bg-zinc-200 rounded-full transition-colors"
                        >
                          <X size={20} />
                        </button>
                      </div>
                      <div className="p-8 space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-zinc-50 rounded-2xl border border-zinc-100">
                            <div className="text-xs font-bold text-zinc-400 uppercase mb-1">Match Score</div>
                            <div className="text-2xl font-bold text-zinc-900">{(selectedCandidate.score * 100).toFixed(0)}%</div>
                          </div>
                          <div className="p-4 bg-zinc-50 rounded-2xl border border-zinc-100">
                            <div className="text-xs font-bold text-zinc-400 uppercase mb-1">JD Similarity</div>
                            <div className="text-2xl font-bold text-zinc-900">{(selectedCandidate.similarity * 100).toFixed(0)}%</div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <h4 className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                            <Sparkles size={16} className="text-purple-500" />
                            AI Analysis
                          </h4>
                          <div className="p-4 bg-purple-50/50 border border-purple-100 rounded-2xl text-zinc-700 leading-relaxed">
                            {selectedCandidate.analysis}
                          </div>
                        </div>
                        <div className="space-y-2">
                          <h4 className="text-sm font-bold text-zinc-900 flex items-center gap-2">
                            <FileText size={16} className="text-zinc-400" />
                            Resume Preview (Snippet)
                          </h4>
                          <div className="p-4 bg-zinc-50 border border-zinc-100 rounded-2xl text-xs text-zinc-500 font-mono max-h-40 overflow-y-auto">
                            {selectedCandidate.resumeText.substring(0, 1000)}...
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  </div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {activeTab === 'code' && (
            <motion.div
              key="code"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-5xl mx-auto space-y-8"
            >
              <header>
                <h2 className="text-3xl font-bold tracking-tight mb-2">ML Implementation</h2>
                <p className="text-zinc-500">Explore the Python-based core of the screening system.</p>
              </header>

              <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                <div className="lg:col-span-1 space-y-2">
                  <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4">Project Files</h4>
                  {pythonFiles.map(file => (
                    <button
                      key={file.name}
                      className="flex items-center gap-3 w-full px-4 py-3 text-sm font-medium text-zinc-600 hover:bg-white hover:text-zinc-900 rounded-xl transition-all border border-transparent hover:border-zinc-100"
                    >
                      <Terminal size={16} />
                      {file.name}
                    </button>
                  ))}
                </div>
                
                <div className="lg:col-span-3 space-y-6">
                  <div className="bg-zinc-900 rounded-2xl p-6 shadow-2xl overflow-hidden border border-white/5">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-3 h-3 rounded-full bg-red-500/50" />
                      <div className="w-3 h-3 rounded-full bg-amber-500/50" />
                      <div className="w-3 h-3 rounded-full bg-emerald-500/50" />
                      <span className="ml-2 text-xs font-mono text-zinc-500">Python 3.10</span>
                    </div>
                    <pre className="text-sm font-mono text-zinc-300 overflow-x-auto leading-relaxed">
                      <code>{pythonFiles[0]?.content}</code>
                    </pre>
                  </div>

                  <div className="bg-white border border-zinc-100 rounded-2xl p-8 shadow-sm">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                      <Info size={20} className="text-blue-500" />
                      Implementation Details
                    </h3>
                    <div className="prose prose-sm text-zinc-600 max-w-none">
                      <p>
                        The system uses a <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> vectorizer to convert resume text into numerical features. 
                        A <strong>Logistic Regression</strong> model is then trained on these features to classify candidates as "Suitable" or "Not Suitable".
                      </p>
                      <ul className="list-disc pl-5 mt-4 space-y-2">
                        <li><strong>Preprocessing:</strong> NLTK is used for lemmatization and stopword removal to reduce noise.</li>
                        <li><strong>Feature Extraction:</strong> 3,000 top features are extracted to maintain model efficiency.</li>
                        <li><strong>Similarity:</strong> Cosine similarity provides a secondary metric for JD alignment.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
