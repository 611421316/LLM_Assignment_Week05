# Yelp Data Translation

This file explains important Yelp JSON fields for CrewAI agents.

## Common fields
- user_id: unique identifier of a user
- item_id: unique identifier of a business or item
- stars: rating score from 1 to 5
- text: review content written by the user

## User-related fields
- review_count: total number of reviews written by the user
- average_stars: average rating the user gives
- useful: number of useful votes received
- funny: number of funny votes received
- cool: number of cool votes received

## Business or item-related understanding
- item review texts help identify strengths, weaknesses, and common sentiment
- repeated positive words often indicate strengths
- repeated negative words often indicate weaknesses

## Interpretation tips
- A user with low average_stars may be stricter
- A user with high average_stars may be more lenient
- Review text tone helps infer writing style
- Frequent praise in item reviews suggests likely strengths
- Frequent complaints in item reviews suggest likely weaknesses