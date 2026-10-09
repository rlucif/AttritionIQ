# AttritionIQ: how it thinks (simple version)

> **In one line:** AttritionIQ guesses who might quit in the next year, explains why, works out whether stepping in is worth the money, and suggests what to do.

---

## The big idea

Losing a good employee is expensive. Gallup estimates that replacing someone costs **half to two times their yearly salary**. The trick is to spot the risk early and act **only where it pays off**.

AttritionIQ follows one chain of logic:

```
Employee data → Chance of leaving → Why? → Is it worth acting? → What to do → How many will we lose? → What does our policy say?
```

---

## How each piece works

### 1. Chance of leaving: learning from past leavers

The app studies 1,470 past employees, of whom about 1 in 6 left. It looks at pay, overtime, promotions, satisfaction, travel, years of experience and more, and learns **which patterns showed up before people quit**. It then gives every employee a score such as "32% chance of leaving within 12 months".

**Why we don't judge it on "accuracy":** since only 1 in 6 people leave, a lazy model that says "nobody will quit" is 84% accurate and completely useless. So we judge it on **how well it finds the people who actually leave**.

**Why the simplest model won:** we tried four methods, from a simple formula to a neural network. The simple one did as well as the fancy ones. When there's a tie, we pick the simple one, because it's easier to explain and to trust.

**Why the scores can be read as real percentages:** raw model scores tend to exaggerate. We adjust them so that "30%" really means about 30 out of 100 such people leave. This matters because the money rule below depends on it.

### 2. Is it worth acting? The money rule

For each person, the app compares two costs:

- **Do nothing:** chance they leave × cost of replacing them
- **Step in:** cost of the retention effort + (chance they leave anyway × cost of replacing them)

It steps in only when that is the cheaper option.

**Example:** say someone's replacement costs 12 months of their salary, the retention effort costs 1 month, and the effort works 4 times out of 10. Stepping in pays off once their chance of leaving is above roughly **21%** (1 ÷ (12 × 0.4)). The salary cancels out, so the same rule works for everyone.

On real test data, the best cut-off came out at **32%**. Below that, we leave people alone. Above it, we flag them.

**Why not just help everyone?** We checked, and helping everyone costs **more than doing nothing**. You'd spend money on lots of people who were never going to leave. **Targeting is where the savings come from.**

With a fixed budget, the app ranks people by "expected saving per unit of money spent" and funds the best-value cases first.

### 3. Why is this person at risk?

For every employee, the app shows the **top reasons pushing their score up or down**, for example "works overtime +, pay below peers +, recently promoted −". A manager sees the *why*, not just a number.

Across everyone, the biggest reasons are **job role, pay, years of experience, overtime, work environment and being paid less than peers**.

### 4. Employee groups: finding natural types

Without telling it who left, the app sorts employees into groups of people who look alike (pay, seniority, overtime, satisfaction and so on). Only **afterwards** do we check each group's quit rate. Three stable groups appear:

| Group | Quit rate | What they look like |
|---|---|---|
| Overtime crew | **34%** | All work overtime, slightly junior |
| Steady core | 12% | No overtime |
| Senior veterans | 8% | Well paid, senior, experienced |

**Takeaway:** what separates the groups is **workload and seniority, not mood**. Satisfaction scores are almost the same across all three.

### 5. What to do: matching fixes to reasons

There are 8 possible actions: cut overtime, review promotion, adjust pay, stock grant, manager 1:1s, training plan, flexible work, and role change. Each one targets certain reasons.

- If someone's top reasons are *overtime* and *poor work-life balance*, "cut overtime" scores highest for them.
- The app also asks: **"How big a pay rise would bring this person below the risk line?"** It tests raises from 0% to 30% and shows the smallest one that works.

There's a second, "people like you" style of recommendation (like Netflix suggestions). It only works once a company has a real record of which actions worked, so for now the reason-matching approach does most of the work.

### 6. Is this person paid fairly?

A separate model predicts what someone **should** earn based on their role, level and experience. It explains about 87% of pay differences. The **gap** between actual and expected pay becomes one of the risk signals. Being underpaid compared with peers raises risk.

### 7. How many people will we lose?

If 10 people each have a 30% chance of leaving, we expect about 3 to leave. The app adds up everyone's chances to get **expected leavers and replacement cost per department**.

It then nudges that number using the **US job-market quit rate**: if people are quitting less nationally, our estimate moves down slightly. We tried a complex forecasting model, but **"next month ≈ this month" predicted better**, so we use the simple one.

### 8. HR assistant: answers from the handbook, not from thin air

When a manager asks "What's our overtime policy?":

1. The app finds the **most relevant sections** of the HR handbook.
2. An AI writes the answer **using only those sections**, so it can't make up policy.
3. With no AI key, it shows a ready-made template answer, so the app still works offline.

It finds the right section first about **8 times out of 10**. It struggles when people use different words, such as "quit" vs "leave".

---

## The ground rules we built in

- 🧪 **Test on people it hasn't seen.** All results come from a group the model never trained on.
- ⚖️ **Simple beats fancy unless fancy is clearly better.** This applied to the risk model and the forecast alike.
- 🤝 **No age, gender or marital status in the scoring.** Adding them improved results by almost nothing, so it isn't worth the ethical risk. We still check that results are fair across those groups.
- 🔎 **Every score is logged** with its reasons, so anyone can later ask "why was this person flagged?"
- 🙋 **A human decides.** The score starts a conversation; it never makes a decision about a person.
- ⚠️ **Linked, not caused.** The reasons show what goes along with quitting, not proof of what causes it.

## Does the logic hold up?

On 294 employees the model had never seen (47 of whom actually left):

- It flagged **36** people, and **19** really left. That's about half, versus 1 in 6 if you picked at random.
- It caught **40%** of all leavers. Most misses were long-serving, well-paid people who didn't show the usual signs.
- Following its advice saved **about 7%** of the cost of people leaving, compared with doing nothing.
- When we changed our cost guesses across 27 different combinations, it still **saved money in 26**.

## What we assumed

- The employee data is a **practice dataset made by IBM**, not real records.
- The cost of a retention effort and how often it works are **our estimates**. The app lets you change them.
- The history of which actions worked is **simulated**, because no real history exists.
- **We wrote** the HR handbook. Its rules come from public sources (EU and UK law, GitLab's public handbook) or are marked as our company's choice.

*Setup and run instructions are in `README.md`. For the detailed version, see `README_Management.md`.*
