import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import base64
from typing import Dict, List, Any
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceFeedbackSystem:
    def __init__(self, huggingface_token: str = None):
        
        self.hf_token = huggingface_token
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        self.headers = {"Authorization": f"Bearer {huggingface_token}"} if huggingface_token else {}
        self.student_data = None
        self.processed_data = None
        self.ai_feedback = None
        
        # Subject ID mapping (you can modify this based on your actual subject IDs)
        self.subject_mapping = {
            "607018ee404ae53194e73d92": "Physics",
            "607018ee404ae53194e73d90": "Chemistry", 
            "607018ee404ae53194e73d91": "Mathematics"
        }
        
    def load_student_data_from_file(self, json_file_path: str):
        """Load student performance data from JSON file."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                self.student_data = json.load(file)
            print(f"✅ Student data loaded successfully from {json_file_path}!")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File {json_file_path} not found.")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Error loading JSON data: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def load_student_data(self, json_data):
        """Load and validate student performance data from JSON."""
        try:
            if isinstance(json_data, str):
                self.student_data = json.loads(json_data)
            elif isinstance(json_data, list):
                # If it's a list, take the first element
                self.student_data = json_data[0] if json_data else {}
            else:
                self.student_data = json_data
            print("✅ Student data loaded successfully!")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Error loading JSON data: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def process_data(self):
        """Process and analyze the student performance data."""
        if not self.student_data:
            raise ValueError("No student data loaded. Call load_student_data() or load_student_data_from_file() first.")
        
        # Handle list input
        data = self.student_data
        if isinstance(data, list):
            data = data[0]
        
        # Extract basic info
        test_info = data.get('test', {})
        subjects_data = data.get('subjects', [])
        sections_data = data.get('sections', [])
        
        
        
        # Initialize processing containers
        processed = {
            'student_info': {
                'name': 'Student',  # Not available in your format
                'test_name': 'QPT Assessment',
                'total_time': test_info.get('totalTime', 0),
                'total_questions': test_info.get('totalQuestions', 0),
                'total_marks': test_info.get('totalMarks', 0)
            },
            'overall_stats': {},
            'subject_performance': {},
            'chapter_performance': {},
            'time_analysis': {},
            'difficulty_analysis': {},
            'concept_mastery': {}
        }
        
        # Overall Statistics
        total_attempted = data.get('totalAttempted', 0)
        total_correct = data.get('totalCorrect', 0)
        total_time_taken = data.get('totalTimeTaken', 0)
        total_marks_scored = data.get('totalMarkScored', 0)
        overall_accuracy = data.get('accuracy', 0)
        
        processed['overall_stats'] = {
            'total_questions': test_info.get('totalQuestions', 0),
            'total_attempted': total_attempted,
            'correct_answers': total_correct,
            'overall_accuracy': round(overall_accuracy, 2),
            'avg_time_per_question': round(total_time_taken / total_attempted, 2) if total_attempted > 0 else 0,
            'total_time_spent': total_time_taken,
            'total_marks_scored': total_marks_scored,
            'total_possible_marks': test_info.get('totalMarks', 0)
        }
        
        # Subject-wise Performance
        subject_performance = {}
        for subject_data in subjects_data:
            subject_id = subject_data.get('subjectId', {}).get('$oid', '')
            subject_name = self.subject_mapping.get(subject_id, f"Subject_{subject_id[:8]}")
            
            subject_performance[subject_name] = {
                'total_attempted': subject_data.get('totalAttempted', 0),
                'correct_answers': subject_data.get('totalCorrect', 0),
                'accuracy': round(subject_data.get('accuracy', 0), 2),
                'avg_time': round(subject_data.get('totalTimeTaken', 0) / subject_data.get('totalAttempted', 1), 2),
                'marks_scored': subject_data.get('totalMarkScored', 0),
                'time_taken': subject_data.get('totalTimeTaken', 0)
            }
        
        processed['subject_performance'] = subject_performance
        
        # Process questions from sections for detailed analysis
        all_questions = []
        chapter_stats = {}
        concept_stats = {}
        difficulty_stats = {}
        
        for section in sections_data:
            section_title = section.get('sectionId', {}).get('title', 'Unknown Section')
            questions = section.get('questions', [])
            
            for question in questions:
                question_data = question.get('questionId', {})
                
                # Extract chapter, topic, concept info
                chapters = question_data.get('chapters', [])
                topics = question_data.get('topics', [])
                concepts = question_data.get('concepts', [])
                
                chapter_name = chapters[0].get('title', 'Unknown Chapter') if chapters else 'Unknown Chapter'
                topic_name = topics[0].get('title', 'Unknown Topic') if topics else 'Unknown Topic'
                concept_name = concepts[0].get('title', 'Unknown Concept') if concepts else 'Unknown Concept'
                
                difficulty = question_data.get('level', 'medium')
                time_taken = question.get('timeTaken', 0)
                status = question.get('status', 'not_attempted')
                
                # Determine if correct
                is_correct = False
                if question.get('markedOptions'):
                    is_correct = question['markedOptions'][0].get('isCorrect', False)
                elif question.get('inputValue'):
                    is_correct = question['inputValue'].get('isCorrect', False)
                
                question_info = {
                    'chapter': chapter_name,
                    'topic': topic_name,
                    'concept': concept_name,
                    'difficulty': difficulty,
                    'time_taken': time_taken,
                    'is_correct': is_correct,
                    'status': status,
                    'section': section_title
                }
                
                all_questions.append(question_info)
                
                # Update chapter stats
                if chapter_name not in chapter_stats:
                    chapter_stats[chapter_name] = {'total': 0, 'correct': 0, 'time': 0}
                chapter_stats[chapter_name]['total'] += 1
                if is_correct:
                    chapter_stats[chapter_name]['correct'] += 1
                chapter_stats[chapter_name]['time'] += time_taken
                
                # Update concept stats
                if concept_name not in concept_stats:
                    concept_stats[concept_name] = {'total': 0, 'correct': 0, 'time': 0}
                concept_stats[concept_name]['total'] += 1
                if is_correct:
                    concept_stats[concept_name]['correct'] += 1
                concept_stats[concept_name]['time'] += time_taken
                
                # Update difficulty stats
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {'total': 0, 'correct': 0, 'time': 0}
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1
                difficulty_stats[difficulty]['time'] += time_taken
        
        # Chapter-wise Performance
        chapter_performance = {}
        for chapter, stats in chapter_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            avg_time = stats['time'] / stats['total'] if stats['total'] > 0 else 0
            
            chapter_performance[chapter] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': round(accuracy, 2),
                'avg_time': round(avg_time, 2),
                'total_time': stats['time']
            }
        
        processed['chapter_performance'] = chapter_performance
        
        # Concept Mastery
        concept_mastery = {}
        for concept, stats in concept_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            avg_time = stats['time'] / stats['total'] if stats['total'] > 0 else 0
            
            concept_mastery[concept] = {
                'questions_attempted': stats['total'],
                'accuracy': round(accuracy, 2),
                'avg_time': round(avg_time, 2),
                'mastery_level': self._determine_mastery_level(accuracy)
            }
        
        processed['concept_mastery'] = concept_mastery
        
        # Difficulty Analysis
        difficulty_analysis = {}
        for difficulty, stats in difficulty_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            avg_time = stats['time'] / stats['total'] if stats['total'] > 0 else 0
            
            difficulty_analysis[difficulty] = {
                'accuracy': round(accuracy, 2),
                'avg_time': round(avg_time, 2),
                'total_questions': stats['total']
            }
        
        processed['difficulty_analysis'] = difficulty_analysis
        
        # Time Analysis
        if all_questions:
            df = pd.DataFrame(all_questions)
            time_values = df['time_taken'].values
            
            processed['time_analysis'] = {
                'avg_time': round(np.mean(time_values), 2),
                'median_time': round(np.median(time_values), 2),
                'min_time': round(np.min(time_values), 2),
                'max_time': round(np.max(time_values), 2),
                'time_distribution': {
                    'fast_questions': len(df[df['time_taken'] < 30]),
                    'medium_questions': len(df[(df['time_taken'] >= 30) & (df['time_taken'] < 120)]),
                    'slow_questions': len(df[df['time_taken'] >= 120])
                }
            }
        
        self.processed_data = processed
        print("✅ Data processing completed!")
        return processed
    
    def _determine_mastery_level(self, accuracy):
        """Determine mastery level based on accuracy."""
        if accuracy >= 85:
            return "Excellent"
        elif accuracy >= 70:
            return "Good"
        elif accuracy >= 50:
            return "Needs Improvement"
        else:
            return "Requires Focus"
    
    def generate_visualizations(self):
        """Generate charts and graphs for the report."""
        if not self.processed_data:
            raise ValueError("Process data first using process_data() method.")
        
        plt.style.use('seaborn-v0_8')
        fig_paths = []
        
        # 1. Subject-wise Performance Chart
        plt.figure(figsize=(12, 8))
        subjects = list(self.processed_data['subject_performance'].keys())
        accuracies = [self.processed_data['subject_performance'][s]['accuracy'] for s in subjects]
        marks = [self.processed_data['subject_performance'][s]['marks_scored'] for s in subjects]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy chart
        bars1 = ax1.bar(subjects, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Subject-wise Accuracy', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        
        for bar, accuracy in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Marks chart
        bars2 = ax2.bar(subjects, marks, color=['#96CEB4', '#FECA57', '#FF9FF3'])
        ax2.set_title('Subject-wise Marks Scored', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Marks Scored', fontsize=12)
        
        for bar, mark in zip(bars2, marks):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{mark}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('subject_performance.png', dpi=300, bbox_inches='tight')
        fig_paths.append('subject_performance.png')
        plt.close()
        
        # 2. Chapter-wise Performance
        if len(self.processed_data['chapter_performance']) > 1:
            plt.figure(figsize=(14, 8))
            chapters = list(self.processed_data['chapter_performance'].keys())
            chapter_accuracies = [self.processed_data['chapter_performance'][c]['accuracy'] for c in chapters]
            
            plt.barh(chapters, chapter_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(chapters))))
            plt.title('Chapter-wise Performance', fontsize=16, fontweight='bold')
            plt.xlabel('Accuracy (%)', fontsize=12)
            plt.xlim(0, 100)
            
            for i, (chapter, accuracy) in enumerate(zip(chapters, chapter_accuracies)):
                plt.text(accuracy + 1, i, f'{accuracy:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('chapter_performance.png', dpi=300, bbox_inches='tight')
            fig_paths.append('chapter_performance.png')
            plt.close()
        
        # 3. Difficulty Analysis
        if self.processed_data['difficulty_analysis']:
            plt.figure(figsize=(10, 6))
            difficulties = list(self.processed_data['difficulty_analysis'].keys())
            diff_accuracies = [self.processed_data['difficulty_analysis'][d]['accuracy'] for d in difficulties]
            
            colors_map = {'easy': '#4CAF50', 'medium': '#FF9800', 'hard': '#F44336'}
            bar_colors = [colors_map.get(d, '#9E9E9E') for d in difficulties]
            
            bars = plt.bar(difficulties, diff_accuracies, color=bar_colors)
            plt.title('Performance by Difficulty Level', fontsize=16, fontweight='bold')
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.xlabel('Difficulty Level', fontsize=12)
            plt.ylim(0, 100)
            
            for bar, accuracy in zip(bars, diff_accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('difficulty_analysis.png', dpi=300, bbox_inches='tight')
            fig_paths.append('difficulty_analysis.png')
            plt.close()
        
        return fig_paths
    
    def generate_ai_feedback(self, model_url="https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"):
        """Generate personalized AI feedback using Hugging Face API."""
        if not self.processed_data:
            raise ValueError("Process data first using process_data() method.")
        
        # Prepare context for AI
        context = self._prepare_ai_context()
        
        # Create the prompt
        prompt = f"""Analyze this student performance data and create an encouraging feedback report:

STUDENT PERFORMANCE SUMMARY:
- Overall Accuracy: {self.processed_data['overall_stats']['overall_accuracy']}%
- Questions Attempted: {self.processed_data['overall_stats']['total_attempted']}
- Correct Answers: {self.processed_data['overall_stats']['correct_answers']}
- Total Time: {self.processed_data['overall_stats']['total_time_spent']/60:.1f} minutes

SUBJECT PERFORMANCE:
{self._format_subject_performance()}

Create a personalized report with:
1. Encouraging introduction
2. Performance highlights
3. Subject-specific insights
4. Time management analysis
5. 3 actionable study recommendations
6. Motivating conclusion

Keep it positive, specific, and helpful."""
        
        try:
            # Try multiple Hugging Face models for better results
            models_to_try = [
                "microsoft/DialoGPT-large",
                "google/flan-t5-large", 
                "bigscience/bloom-560m",
                "microsoft/DialoGPT-medium"
            ]
            
            feedback_generated = False
            
            for model in models_to_try:
                try:
                    api_url = f"https://api-inference.huggingface.co/models/{model}"
                    
                    if "flan-t5" in model:
                        payload = {
                            "inputs": f"Generate student feedback report: {prompt}",
                            "parameters": {
                                "max_new_tokens": 500,
                                "temperature": 0.7,
                                "do_sample": True
                            }
                        }
                    else:
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "max_length": 800,
                                "temperature": 0.7,
                                "do_sample": True,
                                "top_p": 0.9
                            }
                        }
                    
                    response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if isinstance(result, list) and len(result) > 0:
                            if 'generated_text' in result[0]:
                                generated_text = result[0]['generated_text']
                            else:
                                generated_text = str(result[0])
                        elif isinstance(result, dict):
                            generated_text = result.get('generated_text', str(result))
                        else:
                            generated_text = str(result)
                        
                        if generated_text and len(generated_text.strip()) > 100:
                            self.ai_feedback = self._format_ai_response(generated_text, prompt)
                            feedback_generated = True
                            print(f"✅ AI feedback generated successfully using {model}!")
                            break
                        else:
                            print(f"⚠️ {model} returned short response, trying next model...")
                            continue
                    
                    elif response.status_code == 503:
                        print(f"⚠️ {model} is loading, trying next model...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"⚠️ {model} returned status {response.status_code}, trying next model...")
                        continue
                        
                except requests.exceptions.Timeout:
                    print(f"⚠️ {model} timed out, trying next model...")
                    continue
                except Exception as e:
                    print(f"⚠️ Error with {model}: {e}, trying next model...")
                    continue
            
            if not feedback_generated:
                print("⚠️ All AI models failed, generating template-based feedback...")
                self.ai_feedback = self._generate_template_feedback()
            
            return self.ai_feedback
            
        except Exception as e:
            print(f"❌ Error generating AI feedback: {e}")
            print("🔄 Generating template-based feedback as fallback...")
            self.ai_feedback = self._generate_template_feedback()
            return self.ai_feedback
    
    def _format_subject_performance(self):
        """Format subject performance for AI prompt."""
        subject_text = ""
        for subject, data in self.processed_data['subject_performance'].items():
            subject_text += f"- {subject}: {data['accuracy']}% accuracy, {data['marks_scored']} marks\n"
        return subject_text
    
    def _format_ai_response(self, generated_text: str, original_prompt: str) -> str:
        """Format and clean the AI generated response."""
        if original_prompt in generated_text:
            generated_text = generated_text.replace(original_prompt, "").strip()
        
        lines = generated_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                if any(keyword in line.lower() for keyword in ['intro', 'performance', 'time', 'suggestions', 'conclusion']):
                    if not line.startswith('**'):
                        line = f"**{line}**"
                formatted_lines.append(line)
        
        return '\n\n'.join(formatted_lines)
    
    def _generate_template_feedback(self) -> str:
        """Generate template-based feedback when AI fails."""
        overall_accuracy = self.processed_data['overall_stats']['overall_accuracy']
        total_attempted = self.processed_data['overall_stats']['total_attempted']
        total_correct = self.processed_data['overall_stats']['correct_answers']
        total_marks = self.processed_data['overall_stats']['total_marks_scored']
        
        # Find strongest and weakest subjects
        subject_performance = self.processed_data['subject_performance']
        if subject_performance:
            best_subject = max(subject_performance.keys(), key=lambda x: subject_performance[x]['accuracy'])
            weakest_subject = min(subject_performance.keys(), key=lambda x: subject_performance[x]['accuracy'])
        else:
            best_subject = "Mathematics"
            weakest_subject = "Physics"
        
        feedback = f"""**🎉 Congratulations on Your Performance!**

Great job on completing your QPT assessment! You've shown dedication and effort in tackling {total_attempted} questions and achieving {overall_accuracy}% overall accuracy. Your performance demonstrates a solid understanding of the concepts, and there's clear potential for even greater success ahead.

**📊 Performance Highlights**

You've scored {total_marks} marks and correctly answered {total_correct} out of {total_attempted} questions attempted. Here's your subject-wise breakdown:

"""
        
        for subject, data in subject_performance.items():
            feedback += f"• **{subject}**: {data['accuracy']}% accuracy with {data['marks_scored']} marks scored\n"
        
        feedback += f"""
Your strongest performance was in **{best_subject}** - excellent work! This shows you have a solid grasp of the fundamental concepts in this subject.

**⏰ Time Management Analysis**

You spent an average of {self.processed_data['overall_stats']['avg_time_per_question']:.1f} seconds per question, which shows good pacing. """
        
        avg_time = self.processed_data['overall_stats']['avg_time_per_question']
        if avg_time > 120:
            feedback += "Consider practicing with time constraints to improve your speed while maintaining accuracy."
        elif avg_time < 60:
            feedback += "Your quick problem-solving is impressive! Make sure to double-check your work when time permits."
        else:
            feedback += "Your timing is well-balanced between speed and accuracy - keep this up!"
        
        feedback += f"""

**💡 Three Key Recommendations**

1. **Strengthen {weakest_subject}**: Focus extra attention on {weakest_subject} concepts. Dedicate 20-30 minutes daily to practice problems and review fundamental concepts in this subject.

2. **Build on Your {best_subject} Success**: You're doing great in {best_subject}! Use this confidence to tackle more challenging problems and help reinforce your understanding through teaching others.

3. **Balanced Practice Routine**: Maintain a daily study schedule that covers all subjects. Spend more time on weaker areas while keeping your stronger subjects sharp through regular practice.

**🌟 Motivational Conclusion**

Your {overall_accuracy}% accuracy shows you're on the right track! Every question you attempt is a learning opportunity, and your consistent effort will lead to improved performance. Remember, success in competitive exams comes from persistent practice and maintaining a positive attitude.

Keep pushing forward, stay focused on your goals, and trust in your abilities. You have the potential to achieve great things! 🚀

---
*Keep learning, keep growing, and keep believing in yourself!*"""
        
        return feedback
    
    def _prepare_ai_context(self):
        """Prepare formatted context for AI analysis."""
        context = f"""
        TEST INFORMATION:
        - Total Questions: {self.processed_data['student_info']['total_questions']}
        - Total Time Allowed: {self.processed_data['student_info']['total_time']} minutes
        - Total Marks: {self.processed_data['student_info']['total_marks']}
        
        OVERALL PERFORMANCE:
        - Questions Attempted: {self.processed_data['overall_stats']['total_attempted']}
        - Correct Answers: {self.processed_data['overall_stats']['correct_answers']}
        - Overall Accuracy: {self.processed_data['overall_stats']['overall_accuracy']}%
        - Total Time Taken: {self.processed_data['overall_stats']['total_time_spent']/60:.1f} minutes
        - Marks Scored: {self.processed_data['overall_stats']['total_marks_scored']}
        
        SUBJECT-WISE PERFORMANCE:
        """
        
        for subject, data in self.processed_data['subject_performance'].items():
            context += f"""
        - {subject}: {data['accuracy']}% accuracy ({data['correct_answers']} correct out of {data['total_attempted']} attempted)
          Time taken: {data['time_taken']/60:.1f} minutes, Marks: {data['marks_scored']}
        """
        
        if self.processed_data['chapter_performance']:
            context += "\nCHAPTER-WISE PERFORMANCE:\n"
            for chapter, data in self.processed_data['chapter_performance'].items():
                context += f"- {chapter}: {data['accuracy']}% accuracy ({data['correct_answers']}/{data['total_questions']} questions)\n"
        
        return context
    
    def generate_pdf_report(self, filename="student_feedback_report.pdf"):
        """Generate a professional PDF report."""
        if not self.ai_feedback:
            raise ValueError("Generate AI feedback first using generate_ai_feedback() method.")
        
        # Generate visualizations
        chart_paths = self.generate_visualizations()
        
        # Create PDF
        doc = SimpleDocTemplate(filename, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E4057'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4A90A4'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Build the story
        story = []
        
        # Title
        story.append(Paragraph("🎓 QPT Performance Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Test Info Table
        test_info = self.processed_data['student_info']
        test_data = [
            ['Test Name:', test_info.get('test_name', 'QPT Assessment')],
            ['Total Questions:', str(test_info.get('total_questions', 0))],
            ['Total Time:', f"{test_info.get('total_time', 0)} minutes"],
            ['Total Marks:', str(test_info.get('total_marks', 0))],
            ['Report Date:', datetime.now().strftime('%B %d, %Y')]
        ]
        
        test_table = Table(test_data, colWidths=[2*inch, 3*inch])
        test_table.setStyle(TableStyle([
            
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD'))
        ]))
        
        story.append(test_table)
        story.append(Spacer(1, 30))
        
        # Overall Performance Summary
        story.append(Paragraph("📊 Overall Performance Summary", heading_style))
        
        overall_stats = self.processed_data['overall_stats']
        performance_data = [
            ['Metric', 'Value'],
            ['Questions Attempted', f"{overall_stats['total_attempted']}/{overall_stats['total_questions']}"],
            ['Correct Answers', str(overall_stats['correct_answers'])],
            ['Overall Accuracy', f"{overall_stats['overall_accuracy']}%"],
            ['Total Time Spent', f"{overall_stats['total_time_spent']/60:.1f} minutes"],
            ['Average Time per Question', f"{overall_stats['avg_time_per_question']:.1f} seconds"],
            ['Total Marks Scored', f"{overall_stats['total_marks_scored']}/{overall_stats['total_possible_marks']}"]
        ]
        
        performance_table = Table(performance_data, colWidths=[2.5*inch, 2.5*inch])
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90A4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(performance_table)
        story.append(Spacer(1, 20))
        
        # Subject-wise Performance
        story.append(Paragraph("📚 Subject-wise Performance", heading_style))
        
        subject_data = [['Subject', 'Attempted', 'Correct', 'Accuracy', 'Marks', 'Avg Time']]
        for subject, data in self.processed_data['subject_performance'].items():
            subject_data.append([
                subject,
                str(data['total_attempted']),
                str(data['correct_answers']),
                f"{data['accuracy']}%",
                str(data['marks_scored']),
                f"{data['avg_time']:.1f}s"
            ])
        
        subject_table = Table(subject_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        subject_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90A4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(subject_table)
        story.append(Spacer(1, 20))
        
        # Add charts if available
        for chart_path in chart_paths:
            try:
                img = Image(chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"⚠️ Could not add chart {chart_path}: {e}")
        
        # AI Feedback Section
        story.append(PageBreak())
        story.append(Paragraph("🤖 AI-Generated Personalized Feedback", heading_style))
        
        # Split feedback into paragraphs
        feedback_paragraphs = self.ai_feedback.split('\n\n')
        for para in feedback_paragraphs:
            if para.strip():
                # Handle bold text
                if para.startswith('**') and para.endswith('**'):
                    para_style = ParagraphStyle(
                        'BoldPara',
                        parent=normal_style,
                        fontName='Helvetica-Bold',
                        fontSize=12,
                        textColor=colors.HexColor('#2E4057')
                    )
                else:
                    para_style = normal_style
                
                story.append(Paragraph(para.strip(), para_style))
                story.append(Spacer(1, 8))
        
        # Chapter-wise Performance (if available)
        if self.processed_data['chapter_performance']:
            story.append(PageBreak())
            story.append(Paragraph("📖 Chapter-wise Performance Analysis", heading_style))
            
            chapter_data = [['Chapter', 'Questions', 'Correct', 'Accuracy', 'Avg Time']]
            for chapter, data in self.processed_data['chapter_performance'].items():
                chapter_data.append([
                    chapter,
                    str(data['total_questions']),
                    str(data['correct_answers']),
                    f"{data['accuracy']}%",
                    f"{data['avg_time']:.1f}s"
                ])
            
            chapter_table = Table(chapter_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
            chapter_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90A4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(chapter_table)
            story.append(Spacer(1, 20))
        
        # Concept Mastery Analysis
        if self.processed_data['concept_mastery']:
            story.append(Paragraph("🧠 Concept Mastery Analysis", heading_style))
            
            concept_data = [['Concept', 'Questions', 'Accuracy', 'Mastery Level']]
            for concept, data in list(self.processed_data['concept_mastery'].items())[:10]:  # Top 10 concepts
                concept_data.append([
                    concept[:30] + "..." if len(concept) > 30 else concept,
                    str(data['questions_attempted']),
                    f"{data['accuracy']}%",
                    data['mastery_level']
                ])
            
            concept_table = Table(concept_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 1.2*inch])
            concept_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90A4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(concept_table)
            story.append(Spacer(1, 20))
        
        # Time Management Analysis
        story.append(Paragraph("⏰ Time Management Analysis", heading_style))
        
        time_analysis = self.processed_data['time_analysis']
        time_text = f"""
        <b>Average Time per Question:</b> {time_analysis['avg_time']:.1f} seconds<br/>
        <b>Median Time:</b> {time_analysis['median_time']:.1f} seconds<br/>
        <b>Fastest Question:</b> {time_analysis['min_time']:.1f} seconds<br/>
        <b>Slowest Question:</b> {time_analysis['max_time']:.1f} seconds<br/><br/>
        
        <b>Time Distribution:</b><br/>
        • Fast Questions (&lt;30s): {time_analysis['time_distribution']['fast_questions']}<br/>
        • Medium Questions (30-120s): {time_analysis['time_distribution']['medium_questions']}<br/>
        • Slow Questions (&gt;120s): {time_analysis['time_distribution']['slow_questions']}<br/>
        """
        
        story.append(Paragraph(time_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER
        )
        
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", footer_style))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
        story.append(Paragraph("QPT Performance Analysis System", footer_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF report generated successfully: {filename}")
            
            # Clean up chart files
            for chart_path in chart_paths:
                try:
                    import os
                    if os.path.exists(chart_path):
                        os.remove(chart_path)
                except:
                    pass
            
            return filename
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return None
    
    def generate_complete_report(self, json_file_path: str, output_filename: str = None, huggingface_token: str = None):
        """Generate complete analysis report from JSON file."""
        if huggingface_token:
            self.hf_token = huggingface_token
            self.headers = {"Authorization": f"Bearer {huggingface_token}"}
        
        # Load data from file
        if not self.load_student_data_from_file(json_file_path):
            return False
        
        # Process data
        self.process_data()
        
        # Generate AI feedback
        self.generate_ai_feedback()
        
        # Generate PDF report
        if not output_filename:
            output_filename = f"student_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        pdf_path = self.generate_pdf_report(output_filename)
        
        if pdf_path:
            print(f"🎉 Complete report generated successfully!")
            print(f"📄 PDF saved as: {pdf_path}")
            return True
        else:
            print("❌ Failed to generate complete report")
            return False
    
    def display_summary(self):
        """Display a quick summary of the analysis."""
        if not self.processed_data:
            print("❌ No processed data available. Run process_data() first.")
            return
        
        print("\n" + "="*60)
        print("📊 STUDENT PERFORMANCE SUMMARY")
        print("="*60)
        
        overall = self.processed_data['overall_stats']
        print(f"🎯 Overall Accuracy: {overall['overall_accuracy']}%")
        print(f"📝 Questions Attempted: {overall['total_attempted']}/{overall['total_questions']}")
        print(f"✅ Correct Answers: {overall['correct_answers']}")
        print(f"🏆 Total Marks Scored: {overall['total_marks_scored']}/{overall['total_possible_marks']}")
        print(f"⏱️  Average Time per Question: {overall['avg_time_per_question']:.1f} seconds")
        
        print("\n📚 SUBJECT-WISE PERFORMANCE:")
        print("-" * 40)
        for subject, data in self.processed_data['subject_performance'].items():
            print(f"{subject}: {data['accuracy']}% ({data['correct_answers']}/{data['total_attempted']}) - {data['marks_scored']} marks")
        
        if self.processed_data['difficulty_analysis']:
            print("\n🎚️  DIFFICULTY-WISE PERFORMANCE:")
            print("-" * 40)
            for difficulty, data in self.processed_data['difficulty_analysis'].items():
                print(f"{difficulty.capitalize()}: {data['accuracy']}% ({data['total_questions']} questions)")
        
        print("\n" + "="*60)


# Example usage and testing functions
def test_system():
    """Test the system with sample data."""
    
    # Initialize the system
    system = StudentPerformanceFeedbackSystem()
    
    # Test with the provided JSON structure
    sample_json_file = "student_data.json"  # Replace with your actual JSON file path
    
    try:
        print("🚀 Starting Student Performance Analysis...")
        
        # Generate complete report
        success = system.generate_complete_report(
            json_file_path=sample_json_file,
            output_filename="student_performance_report.pdf",
            huggingface_token=None  # Add your Hugging Face token here if available
        )
        
        if success:
            # Display summary
            system.display_summary()
            
            print("\n🎉 Analysis completed successfully!")
            print("📊 Check the generated PDF report for detailed analysis.")
        else:
            print("❌ Analysis failed. Please check the JSON file and try again.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


# Main execution
if __name__ == "__main__":
    # Example usage:
    # 1. Basic usage with JSON file
    system = StudentPerformanceFeedbackSystem()
    
    # Load and process data
    json_file_path = "sample_submission_analysis_1.json"  # Replace with your JSON file path
    
    if system.load_student_data_from_file(json_file_path):
        system.process_data()
        system.generate_ai_feedback()
        system.generate_pdf_report("detailed_student_report.pdf")
        system.display_summary()
    
    # 2. Complete report generation in one step
    # system.generate_complete_report("student_data.json", "complete_report.pdf")
    
    # 3. Test the system
    # test_system()
    
    print("\n📚 System ready for use!")
    print("💡 Usage: system.generate_complete_report('your_file.json', 'output_report.pdf')")
