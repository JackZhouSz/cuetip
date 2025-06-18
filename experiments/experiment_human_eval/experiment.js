// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function(data) {
        window.location.href = 'https://app.prolific.com/submissions/complete?cc=C1C3UV2M'; 
    }
});

// Generate a random user ID
const userId = Math.random().toString(36).substr(2, 9);

// Function to save data
const saveData = async (data) => {
    // Add user ID to the data
    const dataWithUser = {
        ...data,
        userId: userId
    };
    
    try {
        const response = await fetch('save_data.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(dataWithUser)
        });
        const result = await response.json();
        console.log('Data saved successfully:', result);
    } catch (error) {
        console.error('Error saving data:', error);
    }
};

// Function to randomly sample two items from an array
const sampleTwo = (array) => {
    const shuffled = [...array].sort(() => Math.random() - 0.5);
    return [shuffled[0], shuffled[1]];
};

// Welcome screen
const welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class='welcome-container'>
            <h1 class='welcome-title'>Welcome to the Pool Shot Evaluation Study</h1>
            <p class='welcome-description'>In pool, the objective is to score points by knocking your target balls into the table's pockets. Each player has three balls to pot, in this case these are the <strong>red</strong>, <strong>blue</strong>, and <strong>yellow</strong> balls.</p>
            <p class='welcome-description'>You will be shown 20 explanations of shots made in random states of the game. For each explanation, you'll be asked to rate its quality on a scale of 1-7.</p>
            <p class='welcome-description'>1 means the explanation is very poor, and 7 means the explanation is excellent.</p>
            <p class='welcome-description'>You will need to wait 10 seconds before rating each explanation to ensure you have time to fully consider it.</p>
            <p class='welcome-start'><strong>Press any key to begin.</strong></p>
        </div>
    `,
};

// Function to create a trial for a single explanation
const createExplanationTrial = (shotNumber, explanation, sourceNumber) => {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function() {
            return `
                <div class="shot-container">
                    <img src="shot_${shotNumber}.gif" class="shot-gif" />
                    <div class="explanation-container" style="display: flex; justify-content: center; align-items: center; text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
                        <div class="explanation-box">
                            <p>${explanation}</p>
                        </div>
                    </div>
                    <div id="timer" style="text-align: center; margin-top: 20px;">Please wait 10 seconds...</div>
                </div>
            `;
        },
        choices: ['1', '2', '3', '4', '5', '6', '7'],
        button_html: '<button class="jspsych-btn">%choice%</button>',
        prompt: `
            <div style="margin-bottom: 15px;">
                <p style="display: inline-block;">Rate the quality of this explanation (1 = very poor, 7 = excellent)</p>
                <div class="tooltip">
                    <span class="info-button">?</span>
                    <span class="tooltip-text">
                        A high-quality explanation should:
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Accurately describe what happened in the shot</li>
                            <li>Gives specific reasons for its evaluation and is not vague/general</li>
                            <li>Consider both offensive and defensive aspects</li>
                        </ul>
                        Rate 1-3 for explanations that miss most of these aspects<br>
                        Rate 4-5 for explanations that cover some aspects well<br>
                        Rate 6-7 for explanations that thoroughly cover most aspects
                    </span>
                </div>
            </div>
        `,
        trial_duration: null,
        on_load: function() {
            // Disable buttons initially
            document.querySelectorAll('.jspsych-btn').forEach(btn => btn.disabled = true);
            
            // Start 10 second timer
            let timeLeft = 10;
            const timerElement = document.getElementById('timer');
            const timer = setInterval(() => {
                timeLeft--;
                if (timeLeft > 0) {
                    timerElement.textContent = `Please wait ${timeLeft} seconds...`;
                } else {
                    timerElement.textContent = 'You may now rate the explanation';
                    document.querySelectorAll('.jspsych-btn').forEach(btn => btn.disabled = false);
                    clearInterval(timer);
                }
            }, 1000);
        },
        data: {
            task: 'rate_explanation',
            shot_number: shotNumber,
            source_number: sourceNumber,
            explanation: explanation
        },
        on_finish: function(data) {
            saveData({
                task: 'rate_explanation',
                shot_number: shotNumber,
                source_number: sourceNumber,
                explanation: explanation,
                rating: data.response,
                timestamp: Date.now()
            });
        }
    };
};

// Create all trials
let allTrials = [];

// First shuffle the order of shots
const shots = [
    {
        source1: "This shot was a tough but beneficial choice, as it broke up a cluster of balls and maintained a safe position, although it demanded precise aim and English application, increasing its difficulty.",
        source2: "This shot is good because it breaks up clusters of balls, creating opportunities for future shots, and has the potential for multiple balls to be potted in sequence, even though it requires high precision and skill to execute due to the distance and possible need for English.",
        source3: "This shot is a good choice because it successfully pots a target ball, the yellow ball, and has the potential to set us up well for subsequent shots on other target balls like the blue ball, making it a valuable strategic decision despite being somewhat difficult due to factors like distance and the need for precise execution.",
        source4: "This shot is a good one because it successfully pots the high-value yellow ball into the corner pocket, while avoiding contact with lower-value balls like green and brown, making it a great combination of value and skill.",
        source5: "This shot is good because it successfully pots the yellow target ball into the right corner pocket, which is a high-value move, while navigating around the green non-target ball, making it a challenging yet rewarding play.",
        source6: "This shot is good because it successfully pots the yellow ball, one of our targets, into the right corner pocket, while avoiding hitting the green and brown balls, which we don't want to hit yet, showing a good balance between value and difficulty."
    },
    {
        source1: "This shot is good because it successfully pots two target balls, 'yellow' and 'blue', directly addressing our goal of clearing the table of these colors. Although technically demanding due to long distances and large cut angles involved, the payoff in terms of clearing key balls outweighs the risks, showcasing a strong balance between value and difficulty. Additionally, the shot demonstrates strategic foresight by navigating through obstacles and utilizing the table's geometry efficiently, highlighting the player's skill in managing complex scenarios.",
        source2: "This shot has both high value and high difficulty, making it a great but tough shot. By clearing key balls like the yellow and blue, it opens up future opportunities while removing immediate threats, showcasing its strategic importance. However, the execution requires skill due to the distance, angles, and precision needed, highlighting the player's expertise in navigating complex scenarios on the table. In essence, it's a commendable effort that balances risk and reward effectively in the game of pool.",
        source3: "This shot is a good choice because it successfully pots two target balls, the yellow and blue balls, which aligns well with the goal of potting the most target balls possible. Although the shot has extremely high difficulty due to factors like long distance and obstacles, its high value comes from offering a two-way shot possibility, allowing for both offense and defense, and avoiding non-target balls like the green, black, and brown.",
        source4: "This shot is great because we managed to pot two high-value balls, the yellow and blue ones, into their respective corners, making the most of our chances without putting ourselves in a tough spot afterwards, considering the cue ball ended up near the bottom right corner, giving us plenty of options for the next shot.",
        source5: "This shot is great because it results in two target balls being potted, which is very valuable, and although it requires some skill due to the angles involved, particularly when considering the spin imparted on the cue ball from the shot parameters, ultimately the reward outweighs the risk.",
        source6: "This shot is great because we managed to pot two of our target balls, 'yellow' and 'blue', which have high point values, making it a valuable play, and it wasn't too difficult given the angles and positions of the balls and pockets involved."
    },
    {
        source1: "This shot is a good choice because it provides excellent insurance balls and two-way shot possibilities, offering both offensive and defensive advantages, despite being somewhat tricky due to distance and cut angle considerations.",
        source2: "This shot has some great advantages, like being able to pot several target balls and offering good insurance and two-way shot possibilities, but it's also quite tricky due to the long distance and large cut angle involved, making precision crucial.",
        source3: "This shot has its pros and cons. On the plus side, it successfully pots a target ball, the 'blue' one, which is a big advantage. It also sets up potential future shots, making it strategically sound. However, the execution was tricky due to long distances and sharp angles, requiring a lot of skill.",
        source4: "This shot has its pros and cons - on the plus side, we managed to pot the blue ball, but on the downside, we made contact with the black ball, which could've led to trouble if it had gone in, so overall it was a bit of a tricky play.",
        source5: "This shot is good because it allows us to pot the blue ball, one of our targets, into the lower left corner pocket, while avoiding the black ball, which we don't want to pot yet.",
        source6: "This shot has some great advantages, like successfully potting the blue target ball, but it also comes with challenges due to the cue ball having to interact with other balls and cushions along the way, making it harder to control."
    },
    {
        source1: "This shot has a mix of positive and negative points; on the plus side, it offers excellent safety opportunities and avoids potential scratches, making it strategically valuable despite its moderate level of difficulty due to factors like long distance and obstacles. However, the shot's success relies heavily on precision due to the involvement of multiple complexities. In essence, it's a challenging yet potentially rewarding choice, balancing risk against the possibility of gaining a strong defensive position or even potting a target ball directly.",
        source2: "This shot is a good choice because it played well defensively, leaving our opponent in a tricky situation, and managed to avoid complicating the layout further, even though it involved navigating through a challenging obstacle course of other balls over a considerable distance, ultimately resulting in the desired outcome of potting the blue ball.",
        source3: "This shot is considered good despite being quite challenging due to its long distance and obstacles, as it successfully pots a target ball, offering high value through direct achievement and safety opportunities, making it a strategically sound decision.",
        source4: "This shot is good because it successfully pots the blue ball, which is one of our targets, into the corner pocket, while keeping the cue ball from going into a pocket, showing great control over the cue ball's movement after impact.",
        source5: "This shot is good because it successfully hit the blue ball into the pocket, which is one of our target colors, without touching the green or black balls, showing great control over the cue ball's trajectory and speed, making it a high-value and challenging shot.",
        source6: "This shot is good because it successfully pots the blue target ball into the right corner pocket, which can make things tricky, especially for beginners who may not know what \"pot\" means, which is when a ball goes into one of the six pockets around the table."
    },
    {
        source1: "This shot has some great advantages, including offering both offensive and defensive potential, allowing for natural progression from one ball to the next, and minimizing the risk of fouling. However, its difficulty is increased due to long distances and large cut angles involved, making precision crucial. Overall, it's a challenging but potentially rewarding shot.",
        source2: "This shot has some great benefits, like offering two-way shot possibilities and allowing us to pot multiple target balls in sequence, but it's also quite tricky due to the long distance involved and the need for precise speed control, making it a tough call.",
        source3: "This shot is a good choice because it offers a chance to pocket the yellow target ball directly into the right bottom corner pocket, while also setting up a potential opportunity to continue playing offense against the blue target ball afterwards, despite being a relatively long and tricky shot that requires some skill to execute accurately.",
        source4: "This shot is a good one because it successfully pots the high-value yellow ball into the right bottom corner pocket, making it a great choice given its relatively low difficulty due to the clear path to the pocket from the cue ball's starting position near the top cushion.",
        source5: "This shot is a good one because it successfully potted the high-value yellow ball into the right bottom pocket, and considering the speed and angles involved, it wasn't too difficult, making it a great choice given its value.",
        source6: "This shot is a good one because it resulted in potting the yellow target ball into the right bottom corner pocket, which is a high-value move, and it did so without hitting non-target balls like green, black, or brown."
    },
    {
        source1: "This shot has both high value and difficulty due to its ability to pot a target ball directly into a pocket while navigating through a somewhat cluttered table layout, which requires precision in terms of English and speed control, making it a challenging yet potentially rewarding option.",
        source2: "This shot is a great choice because it provides excellent opportunities for clearing multiple target balls and setting up future shots, although it requires precise technique and control over spin and speed, making it somewhat challenging to execute.",
        source3: "This shot is good because it offers extremely high value due to the opportunity to pocket multiple target balls like blue, red, and yellow in makable regions and creates possibilities for future shots, despite being highly difficult due to requirements for precise english and speed control.",
        source4: "This shot is good because it allows us to directly strike the blue ball, which is one of our targets, and send it towards the right top pocket, increasing its chances of being potted, while avoiding hitting non-target balls like the green, black, or brown, thus maximizing the potential score from this single shot.",
        source5: "This shot is great because it successfully pots the blue target ball into the right top corner pocket, which is very valuable, and does so without hitting non-target balls like green, black, or brown.",
        source6: "This shot is great because it results in the blue ball going into the right top pocket after being struck by the cue ball, which shows good control over the cue ball's trajectory."
    },
    {
        source1: "This shot has some great benefits, like creating opportunities to pot multiple target balls in sequence and being near the rail, but it's quite tricky because it requires precise speed control and involves a lot of curve, making it harder to execute; overall, it seems like a decent choice given its moderate value and high difficulty.",
        source2: "This shot is a good choice because it allowed us to pot a target ball ('red'), setting ourselves up well for future shots, despite being a relatively long and tricky shot. By taking advantage of opportunities for multiple-ball positions and combination shots, we increased our chances of success, making this a valuable shot to take.",
        source3: "This shot has its pros and cons, offering opportunities for sequential shots but lacking in terms of safety options, and although it requires skill due to distance and follow requirements, it does provide a chance to clear the table efficiently by targeting multiple balls in sequence.",
        source4: "This shot is good because it successfully pots the red target ball into the right bottom pocket after bouncing off the cushion, showing great control over the cue ball's trajectory, making it a high-value shot with moderate difficulty.",
        source5: "This shot is good because it targets high-value balls like red and has a reasonable chance of success given its moderate speed and accurate aim, making contact with the red ball first before bouncing off the cushion to stop safely.",
        source6: "This shot is a good one because it successfully pots the red target ball into the right bottom pocket, which has high value, and does so without hitting non-target balls like 'green', 'black', or 'brown', making it a relatively safe and easy shot."
    },
    {
        source1: "This shot has its pros and cons, on the plus side it avoids scratches and targets multiple balls, but on the downside it doesn't set us up well for future shots and requires navigating some tricky obstacles, making it a bit of a gamble. In pool, a \"scratch\" refers to when the cue ball goes into a pocket, resulting in a penalty, so avoiding this is always beneficial.",
        source2: "This shot is a great choice because it successfully pots the yellow target ball directly into the right top pocket, avoiding non-target balls like green, black, and brown. Although it requires some skill due to the relatively long distance and the involvement of multiple cushions, the payoff is substantial since it clears a key target ball from the table, aligns well with our goal of minimizing scratch potential, and maximizes the chance of potting the most target balls possible.",
        source3: "This shot is a great example of balancing strategy and skill, as it successfully pockets a target ball, the yellow ball, while navigating a moderately challenging distance and avoiding potential scratches, making it a high-value move despite some difficulties. By executing this shot, the player demonstrates control over the cue ball's trajectory and speed, showcasing proficiency in handling distance and obstacles, key skills in the game of pool.",
        source4: "This shot is good because it successfully pots the yellow target ball into the right top corner pocket, which is a high-value move, while also avoiding contact with non-target balls like green, black, and brown.",
        source5: "This shot is good because it allows us to pot the yellow ball, which is one of our targets, into the right top corner pocket, while avoiding the green, black, and brown balls.",
        source6: "This shot is good because it targets the yellow ball, which is one of our desired balls, and has a high chance of success."
    },
    {
        source1: "Taking this shot is great because it pots the blue target ball directly into the lower left corner pocket, avoiding non-target balls like green, black, and brown, showcasing excellent strategy and skill, despite being quite tricky due to its long distance and obstacles along the way.",
        source2: "This shot is great because it successfully pots the blue target ball while avoiding tricky green, black, and brown balls, making it a highly valuable move according to our assessment of breaking up clusters and creating safety opportunities, despite being quite difficult due to long distances and obstacle balls involved.",
        source3: "This shot has both high value and difficulty due to its ability to pot a target ball while navigating through a crowded table with several obstacles, making it a great example of a challenging yet rewarding shot in pool, known as a \"cluster break\", where the goal is to break apart tightly packed balls.  In pool, the term \"english\" refers to spin imparted on the cue ball, allowing for curved trajectories and better control over the cue ball's movement after striking the object ball. However, the shot requires careful consideration of various factors, including distance, obstacle balls, and rail contact, which increase its difficulty but also make it a valuable learning experience for improving skills in handling complex shots",
        source4: "This shot is good because it successfully pots the blue ball, which is one of our target colors, into the lower left corner pocket, while avoiding the green, black, and brown balls.",
        source5: "This shot is good because it successfully pots the blue ball, hitting it softly enough to avoid scratching, and leaves the cue ball in a safe position, meanwhile avoiding contact with unwanted balls like green, black, and brown.",
        source6: "This shot is good because it successfully pots the blue ball, which is one of our target balls, into the lower left corner pocket, earning us points, and avoids hitting non-target balls like green, black, or brown, making it a high-value shot with moderate difficulty."
    },
    {
        source1: "This shot is a good choice despite its high level of difficulty, mainly because it offers extremely high value in terms of potential outcomes, such as creating opportunities for future shots and leaving your opponent in a tough spot, known as playing safe. In pool, \"playing safe\" refers to hitting a shot that leaves your opponent with no clear opportunity to score, thereby limiting their chances and forcing them to attempt a harder shot.By successfully executing this shot, you not only eliminate a targeted ball but also set yourself up favorably for subsequent turns, either by directly lining up another shot or by complicating your opponent's possible moves.",
        source2: "This shot has both high value and difficulty due to its ability to provide versatile shot options and break up clusters, but requires navigating through obstacles and controlling the cue ball precisely over long distances. In essence, it's a challenging yet potentially rewarding shot that demands skillful execution.",
        source3: "This shot is a good choice because it provides access to \"makable regions\" where multiple balls can be pocketed, offers insurance through easily accessible balls, breaks up clusters, and creates safety opportunities, despite being quite challenging due to long distances, obstacles, and complex trajectories.",
        source4: "This shot is good because it successfully pots the red ball into the top-right corner pocket, which is a high-value move, all while avoiding contact with the brown ball, showing great control over the cue ball's trajectory.",
        source5: "This shot is good because it successfully hits the red ball into the right top corner pocket, avoiding the brown ball, making it valuable without being overly difficult due to the angles involved.",
        source6: "This shot is good because it successfully hits the red ball into the right top pocket, without touching the unwanted brown ball, making it valuable and relatively easy given its trajectory and spin."
    }
];

// First shuffle the order of shots
const shuffledShotIndices = Array.from({length: shots.length}, (_, i) => i).sort(() => Math.random() - 0.5);

shuffledShotIndices.forEach((shotIndex) => {
    const shot = shots[shotIndex];
    
    // Sample one explanation from sources 1-3
    const firstSourceNum = Math.floor(Math.random() * 3) + 1;
    const firstExp = shot[`source${firstSourceNum}`];
    
    // Sample one explanation from sources 4-6
    const secondSourceNum = Math.floor(Math.random() * 3) + 4;
    const secondExp = shot[`source${secondSourceNum}`];
    
    // Create trials for both explanations
    allTrials.push({
        shotNum: shotIndex + 1,
        sourceNum: firstSourceNum,
        explanation: firstExp
    });
    allTrials.push({
        shotNum: shotIndex + 1,
        sourceNum: secondSourceNum,
        explanation: secondExp
    });
});

// Shuffle the order of all explanations
allTrials = allTrials.sort(() => Math.random() - 0.5);

// Add all trials to timeline
let timeline = [welcome];
allTrials.forEach(trial => {
    timeline.push(createExplanationTrial(trial.shotNum, trial.explanation, trial.sourceNum));
});

// Add comment box question
const commentBox = {
    type: jsPsychSurveyText,
    questions: [
        {
            prompt: 'Do you have any additional comments or feedback about the experiment? (Optional)',
            placeholder: 'Type your comments here...',
            rows: 5,
            required: false
        }
    ],
    on_finish: function(data) {
        saveData({
            task: 'comment',
            comment: data.response.Q0,
            timestamp: Date.now()
        });
    }
};

// Experience level question (now at the end)
const experienceQuestion = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <h2>What is your level of experience with pool/billiards?</h2>
        <p>Press the corresponding key:</p>
        <p>1 - None</p>
        <p>2 - Beginner</p>
        <p>3 - Intermediate</p>
        <p>4 - Expert</p>
    `,
    choices: ['1', '2', '3', '4'],
    data: {
        task: 'experience'
    },
    on_finish: function(data) {
        const experienceLevels = ['None', 'Beginner', 'Intermediate', 'Expert'];
        saveData({
            task: 'experience',
            response: experienceLevels[parseInt(data.response) - 1],
            timestamp: Date.now()
        });
    }
};

// Add experience question at the end
timeline.push(experienceQuestion);

// Add comment box before experience question
timeline.push(commentBox);

// Add completion screen
const completion = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <h2>Thank you for participating!</h2>
        <p>The experiment is now complete. Your responses have been saved. Press any key to finish.</p>
    `,
    choices: "ALL_KEYS"
};
timeline.push(completion);

// Start the experiment
jsPsych.run(timeline);
