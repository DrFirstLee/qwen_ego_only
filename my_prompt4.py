def process_image_ego_prompt(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Task:
        Select multiple **precise keypoints** in the image that are essential for performing the action '{action}' on the '{object_name}'.

        🔍 Guidelines:
        - Focus on areas of **human interaction** or **force application** (e.g., handles, grips, pedals)
        - Cover the **entire functional region**, not just one spot  
        - e.g., for a handle: both ends **and** center
        - If multiple '{object_name}' instances are present, mark keypoints on **each of them**
        - Place **at least 3 well-separated** points **within** the object(s)
        - Avoid clustered or irrelevant points

        ⛔ Do NOT:
        - Include any text, labels, or bounding boxes

        ✅ Output format (strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]
        """

def ask_image_ego_prompt(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Task:
        Describe precise regions (in words) that are essential for performing '{action}' on the '{object_name}'.

        🔍 Guidelines:
        - Focus on human-interaction or force-application areas (e.g., handles, grips, pedals, joints, switches, edges used to push/pull).
        - Cover the entire functional region (e.g., both ends and the center of a handle).
        - If multiple '{object_name}' instances are present, describe keypoints on each instance.
        - Be concrete and spatially specific (e.g., "center of the handle", "left hinge", "front-right edge").
        - Start each sentence with the part/region name and, if helpful, mention how it supports '{action}'.
        - Do NOT mention pixel coordinates, sizes, bounding boxes, labels, or add explanations beyond the sentences.

        ✅ Output format (strict):
        - If there is one '{object_name}' in the image,
        [The entire {object_name} area in the center of the image]
        - If there are multiple '{object_name}' in the image,
        [[The center of the {object_name} on the left side of the image], [The tip and butt of the {object_name} on the right side of the image]]
        """


def process_image_exo_with_heatmap_prompt(action, object_name, validation_reason):
    return f"""
    You are given three images:
    1. An **egocentric** image for selecting new keypoints.
    2. A copy of it with previously predicted **incorrect red dots**.
    3. An **exocentric** image showing how the action '{action}' is typically performed on the '{object_name}'.

    📝 Previous feedback:
    "{validation_reason}"

    🎯 Task:
    Select valid [x, y] keypoints in the **first image** that enable proper '{action}' on the '{object_name}'.

    🔍 Use:
    - The second image to avoid past mistakes.
    - The third image to understand correct interaction points.

    📌 Guidelines:
    - Each point must be a single [x, y] within the object
    - Distribute points across functionally relevant areas
    - Avoid clustering or irrelevant placements

    ✅ Output format (strict):
    Return only:
    [
    [x1, y1],
    [x2, y2],
    [x3, y3]
    ]

    ⛔ No extra text, labels, or formatting.
    """



def process_image_exo_prompt(action, object_name):
    return f"""
    You are given two images:
    1. An **egocentric** image where you must select keypoints.
    2. An **exocentric** reference image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    Select multiple [x, y] keypoints in the **egocentric image** that are critical for performing the action '{action}' on the '{object_name}'.

    🔍 Use the exocentric image to:
    - Understand typical interaction patterns
    - Identify functionally important parts (e.g., contact or force areas)

    📌 Guidelines:
    - Keypoints must lie **within** the '{object_name}' in the egocentric image
    - If there are multiple '{object_name}' instances, mark keypoints on **each of them**
    - Place **at least 3 well-separated** points covering the entire functional region
    - e.g., for a handle: both ends and the center
    - Avoid clustering or irrelevant placements

    ⛔ Do NOT:
    - Include text, labels, bounding boxes, or extra formatting

    ✅ Output format (strict):
    [
    [x1, y1],
    [x2, y2],
    [x3, y3]
    ]
    """



def validaton_prompt (action, object_name):
    return f"""
    You are shown three images:
    1. An **egocentric** image for evaluating keypoints.
    2. A copy of it with **red dots** showing selected keypoints.
    3. An **exocentric** image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    Decide whether the red dots are valid for performing the action '{action}' on the '{object_name}'.

    🔍 Use:
    - The red dots image to locate placements.
    - The exocentric image to understand typical interaction points.

    ✅ Valid if:
    - All red dots lie **within** the '{object_name}'
    - Dots cover **functionally relevant** parts (e.g., grips, contact zones)
    - For multiple '{object_name}' instances, dots are reasonably distributed across them

    ❌ Not valid if:
    - Dots are outside or on irrelevant parts
    - Dots are too close together or fail to cover the interactive area

    📢 Output format (strict):
    Return exactly one of:
    - Y: [brief reason]
    - N: [brief reason]

    Examples:
    - Y: Keypoints are correctly placed on functional contact areas.
    - N: Most dots are outside the object or unrelated to the action.

    Do not include any other text.
    """




def validation_and_process_again_prompt(action, object_name):
    return  f"""
    You are an expert in affordance grounding validation.
    Your task is to validate whether the predicted affordance points (shown as red dots in the second image) correctly identify where the action "{action}" can be performed on the "{object_name}".

    Given:
    - First image: The original ego-centric image containing the {object_name}
    - Second image: The ego image with red dots showing where the action "{action}" should be performed
    - Third image: A reference exo-centric image showing the action "{action}" being performed on the {object_name}

    Please analyze the red dots and determine if they correctly identify the affordance points for the action "{action}" on the "{object_name}".

    **Instructions:**
    1. If the red dots correctly identify the affordance points, respond with "Y"
    2. If the red dots are incorrect, respond with "N" and provide a few alternative points where the action should be performed

    **Response Format:**
    - If correct: "Y"
    - If incorrect: "N" followed by new points in the format:
    "N
    New points:
    (x1, y1)
    (x2, y2)
    (x3, y3)
    ..."

    Where (x, y) are pixel coordinates in the original ego image where the action "{action}" should be performed on the "{object_name}".
    The point coordinates you provide must be within the {object_name}.

    **Important:** 
    - Provide coordinates as integers
    - Ensure points are within the image boundaries
    - Points should be where the action can realistically be performed
    - Since it will be reprocessed with Gaussian blur later, please don't judge it too harshly.
    - Consider the object's shape and the nature of the action
    - Base your judgment on the reference exo-centric image
    - Don't repeat similar points too often.

    Your response:
    """