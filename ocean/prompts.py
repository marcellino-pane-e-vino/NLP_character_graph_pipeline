OG_OCEAN_LABELS: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "high openness: curiosity, imagination, exploration, intellectual interest, "
            "openness to new experiences"
        ),
        "negative": (
            "low openness: rigidity, lack of curiosity, conventionality, "
            "resistance to new experiences"
        ),
        "neutral": (
            "no clear evidence about openness, curiosity, imagination, or exploration"
        ),
    },
    "conscientiousness": {
        "positive": (
            "high conscientiousness: carefulness, planning, responsibility, "
            "discipline, persistence, duty"
        ),
        "negative": (
            "low conscientiousness: carelessness, impulsiveness, irresponsibility, "
            "disorganization, lack of planning"
        ),
        "neutral": (
            "no clear evidence about conscientiousness, carefulness, planning, or responsibility"
        ),
    },
    "extraversion": {
        "positive": (
            "high extraversion: sociability, friendliness, outgoing social engagement, "
            "enthusiasm, enjoyment of interaction"
        ),
        "negative": (
            "low extraversion: social withdrawal, reserve, avoidance of interaction, "
            "quietness, lack of social engagement"
        ),
        "neutral": (
            "no clear evidence about extraversion, sociability, assertiveness, or outgoing energy"
        ),
    },
    "agreeableness": {
        "positive": (
            "high agreeableness: kindness, compassion, cooperation, helpfulness, "
            "trust, concern for others"
        ),
        "negative": (
            "low agreeableness: hostility, selfishness, cruelty, refusal to help, "
            "uncooperative behavior"
        ),
        "neutral": (
            "no clear evidence about agreeableness, kindness, compassion, cooperation, or hostility"
        ),
    },
    "neuroticism": {
        "positive": (
            "high neuroticism: fear, anxiety, sadness, distress, emotional instability, worry"
        ),
        "negative": (
            "low neuroticism: calmness, confidence, emotional stability, composure, lack of distress"
        ),
        "neutral": (
            "no clear evidence about neuroticism, fear, anxiety, sadness, worry, or emotional stability"
        ),
    },
}


####################################################################################

OCEAN_LABELS_V2_EVIDENCE_CALIBRATED: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "evidence of high openness: the character shows curiosity, imagination, "
            "intellectual interest, willingness to consider new ideas, tolerance of novelty, "
            "or creative engagement with unfamiliar situations"
        ),
        "negative": (
            "evidence of low openness: the character shows closed-mindedness, rigid thinking, "
            "preference for the familiar, lack of curiosity, conventional judgment, or refusal "
            "to consider new ideas"
        ),
        "neutral": (
            "no reliable evidence for openness: the text does not reveal the character's "
            "curiosity, imagination, intellectual flexibility, creativity, or attitude toward novelty"
        ),
    },
    "conscientiousness": {
        "positive": (
            "evidence of high conscientiousness: the character shows carefulness, planning, "
            "self-control, responsibility, persistence, reliability, duty, or goal-directed effort"
        ),
        "negative": (
            "evidence of low conscientiousness: the character shows carelessness, recklessness, "
            "poor self-control, irresponsibility, unreliability, disorganization, or failure to follow through"
        ),
        "neutral": (
            "no reliable evidence for conscientiousness: the text does not reveal the character's "
            "carefulness, planning, responsibility, reliability, self-control, or persistence"
        ),
    },
    "extraversion": {
        "positive": (
            "evidence of high extraversion: the character actively seeks social interaction, "
            "expresses social confidence, assertiveness, enthusiasm, talkativeness, or enjoyment "
            "of engaging with others"
        ),
        "negative": (
            "evidence of low extraversion: the character avoids social engagement, withdraws from others, "
            "shows social reserve, low assertiveness, reluctance to speak, or preference for solitude"
        ),
        "neutral": (
            "no reliable evidence for extraversion: the text does not reveal the character's "
            "sociability, assertiveness, talkativeness, social energy, or preference about interaction"
        ),
    },
    "agreeableness": {
        "positive": (
            "evidence of high agreeableness: the character shows kindness, empathy, cooperation, "
            "patience, trust, forgiveness, helpfulness, concern for others, or willingness to reduce conflict"
        ),
        "negative": (
            "evidence of low agreeableness: the character shows hostility, selfishness, suspicion, "
            "coldness, competitiveness, cruelty, manipulation, refusal to cooperate, or disregard for others"
        ),
        "neutral": (
            "no reliable evidence for agreeableness: the text does not reveal the character's "
            "kindness, empathy, cooperation, trust, hostility, selfishness, or concern for others"
        ),
    },
    "neuroticism": {
        "positive": (
            "evidence of high neuroticism: the character shows anxiety, fearfulness, insecurity, "
            "emotional volatility, distress, worry, shame, panic, or difficulty staying composed"
        ),
        "negative": (
            "evidence of low neuroticism: the character shows calmness, emotional stability, "
            "confidence, resilience, composure, steadiness under pressure, or quick recovery from distress"
        ),
        "neutral": (
            "no reliable evidence for neuroticism: the text does not reveal the character's "
            "anxiety, fearfulness, emotional stability, distress, confidence, or composure"
        ),
    },
}

###########################################################################################

OCEAN_LABELS_V2_CONSERVATIVE: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "clear personality evidence of high openness: the character displays a meaningful tendency "
            "toward curiosity, imagination, creative thought, intellectual exploration, or acceptance of unfamiliar ideas"
        ),
        "negative": (
            "clear personality evidence of low openness: the character displays a meaningful tendency "
            "toward rigid judgment, closed-mindedness, conventional thinking, or rejection of unfamiliar ideas"
        ),
        "neutral": (
            "insufficient personality evidence for openness: the text may describe events, movement, setting, "
            "or dialogue, but it does not clearly support an openness inference"
        ),
    },
    "conscientiousness": {
        "positive": (
            "clear personality evidence of high conscientiousness: the character displays a meaningful tendency "
            "toward planning, discipline, responsibility, reliability, careful action, or persistence"
        ),
        "negative": (
            "clear personality evidence of low conscientiousness: the character displays a meaningful tendency "
            "toward carelessness, irresponsibility, impulsive action, unreliability, disorganization, or lack of follow-through"
        ),
        "neutral": (
            "insufficient personality evidence for conscientiousness: the text may describe events, movement, setting, "
            "or dialogue, but it does not clearly support a conscientiousness inference"
        ),
    },
    "extraversion": {
        "positive": (
            "clear personality evidence of high extraversion: the character displays a meaningful tendency "
            "toward social initiative, assertive interaction, talkativeness, enthusiasm, or enjoyment of social engagement"
        ),
        "negative": (
            "clear personality evidence of low extraversion: the character displays a meaningful tendency "
            "toward social withdrawal, reserve, avoidance of interaction, low assertiveness, or preference for solitude"
        ),
        "neutral": (
            "insufficient personality evidence for extraversion: the text may describe events, movement, setting, "
            "or dialogue, but it does not clearly support an extraversion inference"
        ),
    },
    "agreeableness": {
        "positive": (
            "clear personality evidence of high agreeableness: the character displays a meaningful tendency "
            "toward empathy, kindness, cooperation, forgiveness, trust, helpfulness, or concern for others"
        ),
        "negative": (
            "clear personality evidence of low agreeableness: the character displays a meaningful tendency "
            "toward hostility, selfishness, suspicion, coldness, manipulation, conflict, or disregard for others"
        ),
        "neutral": (
            "insufficient personality evidence for agreeableness: the text may describe events, movement, setting, "
            "or dialogue, but it does not clearly support an agreeableness inference"
        ),
    },
    "neuroticism": {
        "positive": (
            "clear personality evidence of high neuroticism: the character displays a meaningful tendency "
            "toward worry, insecurity, emotional instability, fearfulness, distress, panic, or difficulty staying composed"
        ),
        "negative": (
            "clear personality evidence of low neuroticism: the character displays a meaningful tendency "
            "toward calmness, confidence, emotional stability, resilience, composure, or steadiness under pressure"
        ),
        "neutral": (
            "insufficient personality evidence for neuroticism: the text may describe events, movement, setting, "
            "or dialogue, but it does not clearly support a neuroticism inference"
        ),
    },
}

###########################################################################################

OCEAN_LABELS_V2_NARRATIVE_INFERENCE: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "narrative evidence of high openness: through action, dialogue, thought, reaction, or choice, "
            "the character appears curious, imaginative, intellectually flexible, creative, exploratory, "
            "or receptive to unfamiliar experiences and ideas"
        ),
        "negative": (
            "narrative evidence of low openness: through action, dialogue, thought, reaction, or choice, "
            "the character appears rigid, incurious, conventional, dismissive of unfamiliar ideas, "
            "or strongly attached to familiar ways of thinking"
        ),
        "neutral": (
            "no usable narrative evidence for openness: the passage does not show how the character thinks about, "
            "reacts to, or evaluates novelty, ideas, imagination, creativity, or unfamiliar experience"
        ),
    },
    "conscientiousness": {
        "positive": (
            "narrative evidence of high conscientiousness: through action, dialogue, thought, reaction, or choice, "
            "the character appears careful, dutiful, disciplined, persistent, responsible, reliable, "
            "or deliberate in pursuing goals"
        ),
        "negative": (
            "narrative evidence of low conscientiousness: through action, dialogue, thought, reaction, or choice, "
            "the character appears careless, reckless, impulsive, unreliable, irresponsible, disorganized, "
            "or unwilling to persist"
        ),
        "neutral": (
            "no usable narrative evidence for conscientiousness: the passage does not show how careful, reliable, "
            "responsible, disciplined, persistent, or impulsive the character is"
        ),
    },
    "extraversion": {
        "positive": (
            "narrative evidence of high extraversion: through action, dialogue, thought, reaction, or choice, "
            "the character appears socially energetic, expressive, assertive, talkative, enthusiastic, "
            "or eager to engage with others"
        ),
        "negative": (
            "narrative evidence of low extraversion: through action, dialogue, thought, reaction, or choice, "
            "the character appears socially withdrawn, reserved, quiet by preference, reluctant to engage, "
            "low in assertiveness, or oriented toward solitude"
        ),
        "neutral": (
            "no usable narrative evidence for extraversion: the passage does not show the character's social energy, "
            "assertiveness, talkativeness, expressiveness, withdrawal, or preference for interaction"
        ),
    },
    "agreeableness": {
        "positive": (
            "narrative evidence of high agreeableness: through action, dialogue, thought, reaction, or choice, "
            "the character appears kind, empathetic, cooperative, forgiving, trusting, helpful, patient, "
            "or concerned for another being"
        ),
        "negative": (
            "narrative evidence of low agreeableness: through action, dialogue, thought, reaction, or choice, "
            "the character appears hostile, selfish, suspicious, cold, manipulative, cruel, uncooperative, "
            "or unconcerned with another being"
        ),
        "neutral": (
            "no usable narrative evidence for agreeableness: the passage does not show the character's kindness, "
            "empathy, cooperation, trust, hostility, selfishness, or concern for others"
        ),
    },
    "neuroticism": {
        "positive": (
            "narrative evidence of high neuroticism: through action, dialogue, thought, reaction, or choice, "
            "the character appears anxious, fearful, ashamed, insecure, emotionally unstable, distressed, "
    "worried, panicked, or unable to remain composed"
            ),
            "negative": (
                "narrative evidence of low neuroticism: through action, dialogue, thought, reaction, or choice, "
                "the character appears calm, confident, emotionally stable, composed, resilient, steady, "
                "or able to recover from distress"
            ),
            "neutral": (
                "no usable narrative evidence for neuroticism: the passage does not show the character's anxiety, "
                "fear, distress, emotional stability, confidence, composure, or resilience"
            ),
        },
    }
###########################################################################################



###########################################################################################



###########################################################################################



###########################################################################################


