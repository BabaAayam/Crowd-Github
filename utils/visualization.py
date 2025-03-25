import cv2

def visualize_frame(frame, count, boxes, scores, classes, height, width):
    """Draw bounding boxes and count"""
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 0:  # Person class
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * width), int(ymin * height)
            x2, y2 = int(xmax * width), int(ymax * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f"People: {count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame