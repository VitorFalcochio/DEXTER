import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image.")
    parser.add_argument("--id", type=int, default=0, help="Marker ID")
    parser.add_argument("--size", type=int, default=700, help="Marker size in pixels")
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_4X4_50",
        help="Dictionary name from cv2.aruco (example: DICT_4X4_50)",
    )
    parser.add_argument("--out", type=str, default="aruco_id0.png", help="Output PNG path")
    args = parser.parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not available in this environment.")

    if not hasattr(cv2.aruco, args.dict):
        raise ValueError(f"Dictionary '{args.dict}' not found in cv2.aruco.")

    dict_id = getattr(cv2.aruco, args.dict)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    marker = cv2.aruco.generateImageMarker(dictionary, args.id, args.size)
    cv2.imwrite(args.out, marker)
    print(f"Marker generated: {args.out} (id={args.id}, dict={args.dict}, size={args.size}px)")


if __name__ == "__main__":
    main()
