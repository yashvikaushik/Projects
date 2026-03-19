const app = {
    state: {
        domains: [],
        currentDomainId: null,
        sequence: [],
        currentIndex: 0
    },

    async init() {
        this.bindEvents();
        await this.loadDomains();
    },

    bindEvents() {
        document.getElementById('btn-restart').addEventListener('click', () => this.resetSession());
        document.getElementById('analyze-btn').addEventListener('click', () => this.analyzeAnswer());
        document.getElementById('btn-next-question').addEventListener('click', () => this.advanceNextQuestion());
    },

    async loadDomains() {
        const res = await fetch('/api/domains');
        this.state.domains = await res.json();
        
        const grid = document.getElementById('domain-grid');
        grid.innerHTML = this.state.domains.map(d => `
            <div class="domain-card" onclick="app.startDomain('${d.id}')">
                <h2>${d.name}</h2>
                <p>${d.count} Questions</p>
            </div>
        `).join('');

        document.getElementById('view-domains').classList.remove('hidden');
        document.getElementById('view-flow').classList.add('hidden');
    },

    async startDomain(domainId) {
        this.state.currentDomainId = domainId;
        document.getElementById('current-domain-label').innerText = domainId;
        
        const res = await fetch(`/api/questions/${encodeURIComponent(domainId)}`);
        this.state.sequence = await res.json();
        this.state.currentIndex = 0;

        document.getElementById('view-domains').classList.add('hidden');
        document.getElementById('view-flow').classList.remove('hidden');
        
        this.loadCurrentQuestion();
    },

    loadCurrentQuestion() {
        if (this.state.currentIndex >= this.state.sequence.length) {
            this.showCompletionState();
            return;
        }

        const q = this.state.sequence[this.state.currentIndex];
        
        const currentNum = this.state.currentIndex + 1;
        const total = this.state.sequence.length;
        document.getElementById('progress-text').innerText = `Question ${currentNum} of ${total}`;
        const pct = (currentNum / total) * 100;
        document.getElementById('progress-fill').style.width = `${pct}%`;

        document.getElementById('current-question-text').innerText = q.question;
        document.getElementById('answer-text').value = '';
        document.getElementById('answer-text').disabled = false;
        
        document.getElementById('empty-state').classList.remove('hidden');
        document.getElementById('results-state').classList.add('hidden');
        document.getElementById('completion-state').classList.add('hidden');
    },

    async analyzeAnswer() {
        if (this.state.currentIndex >= this.state.sequence.length) return;
        
        const qId = this.state.sequence[this.state.currentIndex].id;
        const answer = document.getElementById('answer-text').value;

        if (!answer.trim()) return;

        const btn = document.getElementById('analyze-btn');
        btn.innerText = 'Analyzing...';
        btn.disabled = true;

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question_id: qId, answer: answer })
            });

            if (res.ok) {
                const data = await res.json();
                this.renderResults(data);
                
                // Freeze the input text area after explicit submission
                document.getElementById('answer-text').disabled = true;
            }
        } catch (e) {
            console.error("Analysis failed:", e);
        } finally {
            btn.innerText = 'Analyze ⚡';
            btn.disabled = false;
        }
    },

    renderResults(data) {
        document.getElementById('empty-state').classList.add('hidden');
        const resultsState = document.getElementById('results-state');
        resultsState.classList.remove('hidden');

        // Render Score
        const scoreValDiv = document.getElementById('score-text');
        this.animateValue(scoreValDiv, parseInt(scoreValDiv.innerText) || 0, data.score, 1000);
        
        const circle = document.querySelector('.progress-ring__circle');
        const radius = circle.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        const offset = circumference - (data.score / 100) * circumference;
        
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        setTimeout(() => { circle.style.strokeDashoffset = offset; }, 50);

        let color = 'var(--accent)';
        if(data.score < 50) color = 'var(--secondary)';
        else if(data.score > 80) color = 'var(--success)';
        circle.style.stroke = color;

        let label = "Needs Work";
        if (data.score > 80) label = "Excellent Match";
        else if (data.score > 50) label = "Good Effort";
        
        document.getElementById('score-label').innerText = label;
        document.getElementById('score-label').style.color = color;

        document.getElementById('feedback-list').innerHTML = data.feedback.map(f => `<li>${f}</li>`).join('');

        const missingHtml = data.missing_concepts.map(kw => `<span class="tag">${kw}</span>`);
        const tagsContainer = document.getElementById('missing-tags');
        if (data.missing_concepts.length === 0) {
            tagsContainer.innerHTML = `<span class="tag matched">Perfect coverage!</span>`;
        } else {
            tagsContainer.innerHTML = missingHtml.join('');
        }

        document.getElementById('ideal-answer-text').innerText = data.ideal_answer;
    },

    advanceNextQuestion() {
        this.state.currentIndex++;
        this.loadCurrentQuestion();
    },

    showCompletionState() {
        document.getElementById('empty-state').classList.add('hidden');
        document.getElementById('results-state').classList.add('hidden');
        
        document.getElementById('completion-state').classList.remove('hidden');
        document.getElementById('current-question-text').innerText = "All questions completed!";
        document.getElementById('answer-text').disabled = true;
    },

    resetSession() {
        this.loadDomains();
    },

    animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());
