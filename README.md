# Oral Language Learning Interactive Assistant System

**National Tsing Hua University EE Final Project - 113061529 Jie-Xiang Yang**

> A multimodal language learning platform based on Whisper and Qwen2-Audio, providing personalized pronunciation evaluation and real-time dialogue practice.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lOXTTMYY521QtAjKdNYNB6YN6sXxkug0?usp=sharing)

## Table of Contents

- [Author's Foreword](#authors-foreword)
- [Project Highlights](#project-highlights)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Module Overview](#module-overview)
- [User Guide](#user-guide)

## Author's Foreword

This repository contains the final project for the *TAICA Generative AI: Principles and Practices of Text and Image Generation* joint university course. The project implements an oral language practice platform that utilizes Audio-LLMs to analyze user recordings, evaluate pronunciation, and facilitate dialogue practice. 

Please note that the Qwen2-Audio-7B model requires dedicated hardware and cannot be run on the free tier of Google Colab. A local environment with sufficient GPU resources is necessary for full functionality. 

## Project Highlights

### Multimodal AI Architecture
- **Whisper Speech Recognition:** OpenAI's state-of-the-art speech-to-text model.
- **Qwen2-Audio Analysis:** Alibaba's multimodal language model for direct audio comprehension.
- **Hardware Fallback Mechanism:** Automatically switches to a CPU-based simplified mode when GPU memory is insufficient.

### Personalized Learning Experience
- **5-Tier Difficulty System:** Ranges from Beginner (TOEIC 250) to Advanced (TOEIC 905+).
- **6 Scenario Simulations:** Airport, Restaurant, Interview, Socializing, Medical, and Academic.
- **Real-time Assessment:** Comprehensive evaluation of pronunciation accuracy and fluency.
- **Adaptive Feedback:** Personalized suggestions based on the learner's proficiency level.

### Resource Management
- **Dynamic Memory Monitoring:** Real-time GPU/CPU usage tracking.
- **Adaptive Model Loading:** Optimizes based on available hardware specifications.
- **Emergency Clearing Mechanism:** Built-in safeguards to prevent out-of-memory (OOM) errors.

### Modern UI Design
- **Responsive Interface:** Optimized for both desktop and mobile devices.
- **Glassmorphism Design:** Clean, modern visual aesthetics.
- **Accessibility:** Designed for an intuitive user experience.

## Core Features

### Dual-Mode Learning System

**Preset Scenario Dialogues**
- Airport: Customs, boarding, passport control.
- Restaurant: Menu inquiries, ordering, checkout.
- Job Interview: Q&A and self-introductions.
- Daily Socializing: Greetings, small talk, interactions.
- Medical Consultation: Symptom description, medical communication.
- Academic Discussion: Classroom participation, seminar interactions.

**Free Conversation Mode**
- User-defined scenarios and topics for open-ended practice.

### Multi-level Speech Analysis

**Basic Analysis (Simplified Mode)**
- Speech recognition accuracy evaluation.
- Fundamental pronunciation scoring algorithm.
- Fluency statistical analysis.

**Advanced Analysis (Audio-LLM Mode)**
- Direct audio content comprehension.
- Context-aware feedback.
- Detailed pronunciation correction suggestions.

### Personalized Difficulty System

| Level | TOEIC Score | Evaluation Criteria | Feedback Style |
|---|---|---|---|
| Beginner | 250-400 | Basic pronunciation clarity | Highly encouraging (+15 score adjustment) |
| Elementary | 405-600 | Basic conversational fluency | Encouraging (+10 score adjustment) |
| Intermediate | 605-780 | Grammatical accuracy and naturalness | Balanced (Standard scoring) |
| Upper-Intermediate | 785-900 | Idiom usage and detailed pronunciation | Constructive (-5 score adjustment) |
| Advanced | 905+ | Professional-level fluency | Detailed analysis (-10 score adjustment) |

### Advanced Settings

- **Pronunciation Focus:** Target specific consonants, vowels, liaisons, stress patterns, intonation, and rhythm.
- **Accent Preferences:** Choose between General American, Received Pronunciation (British), or a flexible mode.
- **Learning Tracking:** Auto-saves practice history, analyzes improvement trends, and supports data export.

## System Architecture

```mermaid
graph TB
    A[User Audio Input] --> B[Whisper Speech Recognition]
    B --> C{Memory Check}
    C -->|Sufficient| D[Qwen2-Audio Analysis]
    C -->|Insufficient| E[Simplified Analysis Mode]
    D --> F[Detailed Pronunciation Feedback]
    E --> G[Basic Pronunciation Scoring]
    F --> H[Personalized Response Generation]
    G --> H
    H --> I[User Interface Display]
    
    J[Memory Monitor] --> C
    K[Difficulty Config System] --> F
    K --> G
    L[Scenario Manager] --> H
