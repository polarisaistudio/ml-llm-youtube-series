# Video 3: When to Use Traditional ML vs Modern AI - Decision Framework
**Target Length:** 10 minutes  
**Focus:** Practical business decision-making  
**Audience:** Beginners who need to choose the right approach

## Opening (0:00-0:30)
"You've seen the concepts, you've seen the code. Now comes the most important question: How do you decide which approach to use for YOUR specific problem?

I'm going to give you a simple decision framework that I use in consulting, plus real examples from companies I've worked with. By the end, you'll know exactly which tool to reach for."

**[Visual: Split decision path diagram]**

## Section 1: The Decision Framework (0:30-2:30)
"Here's my 4-question framework. Answer these honestly about your project:

**[Visual: Decision tree flowchart appearing step by step]**

**Question 1: What type of data do you have?**
- Structured (spreadsheet-friendly) → Lean Traditional ML
- Unstructured (text, images, audio) → Lean Modern AI

**Question 2: How much data do you have?**
- Less than 10,000 samples → Traditional ML or pre-trained Modern AI
- More than 100,000 samples → Either approach works
- Millions of samples → Modern AI advantage

**Question 3: Do you need to explain decisions?**
- Yes (legal, medical, financial) → Traditional ML required
- No (internal optimization, recommendations) → Either works

**Question 4: What's your budget?**
- Limited ($100s/month) → Traditional ML
- Flexible ($1000s/month) → Modern AI possible

Let me show you how this works with real scenarios."

## Section 2: Real Business Scenario 1 - Bank Loan Approval (2:30-4:00)
"**Scenario:** A bank needs to automate loan approvals.

**[Visual: Bank loan application form]**

Let's apply our framework:

**Question 1 - Data Type:** 
Structured - income, credit score, employment history, debt-to-income ratio
✓ **Point to Traditional ML**

**Question 2 - Data Volume:**
They have 50,000 historical loan applications
✓ **Either approach works**

**Question 3 - Explainability:**
Regulators require them to explain every denial: 'You were denied because your debt-to-income ratio exceeds 40%'
✓ **Traditional ML required**

**Question 4 - Budget:**  
Need to process thousands of applications daily at low cost
✓ **Traditional ML advantage**

**Decision: Traditional ML**

**[Visual: Simple loan approval algorithm]**

**Real outcome:** I helped implement this with XGBoost. Processes 1000 applications per day at $50/month server cost, with full explainability for regulators."

## Section 3: Real Business Scenario 2 - Customer Service Chatbot (4:00-5:30)
"**Scenario:** E-commerce company wants to automate customer support.

**[Visual: Chat interface with customer messages]**

Framework application:

**Question 1 - Data Type:**
Unstructured - natural language conversations, varied topics, emotional context
✓ **Point to Modern AI**

**Question 2 - Data Volume:**
They have 100,000 past support conversations
✓ **Either approach works, but Modern AI better for language**

**Question 3 - Explainability:**
Internal tool - they care more about customer satisfaction than explaining every response
✓ **Modern AI acceptable**

**Question 4 - Budget:**
$500/month budget for customer support automation
✓ **Modern AI feasible**

**Decision: Modern AI**

**[Visual: GPT-powered chatbot interface]**

**Real outcome:** Implemented GPT-3.5 integration. Handles 70% of inquiries automatically, customer satisfaction increased 15%, costs $300/month in API fees."

## Section 4: Real Business Scenario 3 - Netflix-Style Hybrid (5:30-7:00)
"**Scenario:** Streaming service wants to improve recommendations.

**[Visual: Netflix-like interface]**

This is where it gets interesting - they need BOTH approaches:

**For User Preferences (Traditional ML):**
- Data: Structured viewing history, ratings, demographics
- Volume: Millions of users, billions of interactions  
- Explainability: Helpful for 'Why was this recommended?'
- Budget: Needs to be cost-effective at scale

**Decision: Traditional ML for core recommendations**

**For Personalized Thumbnails (Modern AI):**
- Data: Images, user preferences, contextual information
- Volume: Need to generate thousands of variations
- Explainability: Not critical for thumbnails
- Budget: Higher value feature, worth the cost

**Decision: Modern AI for thumbnail generation**

**[Visual: Same movie with different thumbnails for different users]**

**Real outcome:** Most successful recommendation systems use this hybrid approach. Traditional ML for the heavy lifting, Modern AI for the creative/complex parts."

## Section 5: Common Decision Mistakes (7:00-8:30)
"Here are the mistakes I see most often:

**[Visual: Red X over each mistake]**

**Mistake 1: Using Modern AI for everything because it's trendy**
- Client wanted to use GPT for sales forecasting
- Traditional time series analysis was 10x cheaper and more accurate
- **Rule:** Don't use a Ferrari to deliver pizza

**Mistake 2: Sticking with Traditional ML when you need Modern AI capabilities**  
- Company tried to build sentiment analysis with keyword matching
- Couldn't handle sarcasm, context, or nuanced language
- **Rule:** If humans need context to understand it, you probably need Modern AI

**Mistake 3: Not considering the total cost of ownership**
- Modern AI model costs $0.001 per prediction
- Seems cheap until you're doing 1 million predictions daily ($1000/day)
- Traditional ML: high upfront cost, near-zero marginal cost
- **Rule:** Calculate costs at your expected scale

**Mistake 4: Ignoring regulatory requirements**
- Healthcare AI needs to explain treatment recommendations
- Modern AI 'black box' isn't legally acceptable
- **Rule:** Check compliance requirements first, not last"

## Section 6: Quick Decision Flowchart (8:30-9:30)
"Let me give you a rapid-fire decision tool:

**[Visual: Animated flowchart]**

**START HERE:**

**Is your data in spreadsheets/databases?**
- Yes → Traditional ML likely
- No → Continue

**Do you need to explain every decision?**  
- Yes → Traditional ML required
- No → Continue

**Are you working with text, images, or audio?**
- Yes → Modern AI likely
- No → Traditional ML likely

**Do you have budget for $100+/month in computing costs?**
- No → Traditional ML or free pre-trained models
- Yes → Modern AI possible

**Quick reality check:**
- More than 80% of business problems → Traditional ML
- Text/image/audio problems → Modern AI
- Creative/generative tasks → Modern AI only
- When in doubt → Start with Traditional ML, upgrade if needed"

## Closing & Next Video (9:30-10:00)
"The key insight: Most successful AI projects use the simplest approach that works. Don't optimize for impressiveness - optimize for solving the actual problem.

In our final video of this series, I'll give you a realistic roadmap for learning both approaches, including the biggest mistakes beginners make and how to avoid them.

What's your specific use case? Drop it in the comments and I'll tell you which approach I'd recommend. 

Subscribe to see the learning roadmap - it might surprise you how different it is from the '30 days to AI expert' courses you see everywhere."

**[End screen with comment engagement and subscribe button]**

---

## Production Notes:

**Visual Elements Needed:**
- Decision framework flowchart (animated)
- Real application screenshots (bank forms, chatbots, Netflix interface)
- Cost comparison charts  
- Before/after results from case studies
- Mistake illustrations (red X animations)

**Case Study Details:**
- Use anonymized but realistic examples
- Include specific metrics and outcomes
- Show actual code snippets where appropriate
- Mention tools used (XGBoost, GPT-3.5, etc.)

**Engagement Strategy:**
- Ask for viewer's use cases in comments
- Offer specific recommendations
- Build anticipation for learning roadmap video
- Include practical decision templates in description

**Supplementary Materials:**
- Decision framework PDF download
- Use case calculator spreadsheet  
- Links to mentioned tools and services
- Cost estimation templates